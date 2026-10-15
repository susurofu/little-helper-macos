import json
import logging

from pathlib import Path
from threading import Lock, Thread

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.llm_settings import model_catalog, hardware_info, model_details, bounded_fragments

from backend.ollama_engine import OllamaEngine, PromptPreparator
from backend.db_constructor import ChromaDBManager
from backend.corpus_parser import CorpusFolderIterator


app = FastAPI()


# Enable CORS for the browser extension.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Project paths.
PROJECT_DIR = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_DIR / "global_config.json"

DB_PATH = PROJECT_DIR / "database"


# Load global configuration.
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        f"Configuration file not found: {CONFIG_PATH}"
    )

with CONFIG_PATH.open("r", encoding="utf-8") as file:
    global_config = json.load(file)


# Application state.
app.state.prompt = None
app.state.prompt_being_processed = None

app.state.rag_lock = Lock()
app.state.config_lock = Lock()


def initial_rag_status(status="idle"):
    """
    Return the initial RAG indexing status.
    """

    return {
        "status": status,
        "total_files": 0,
        "processed_files": 0,
        "db_size": 0,
        "updated_files": 0,
        "unchanged_files": 0,
        "empty_files": 0,
        "failed_files": 0,
        "removed_files": 0,
        "current_file": None,
        "last_file_error": None,
        "error": None,
    }


app.state.rag_status = initial_rag_status()


# Request models.
class PromptRequest(BaseModel):
    prompt: str = Field(min_length=1)
    rag_enabled: bool = False


class PathRequest(BaseModel):
    rag_path: str


def current_rag_folder():
    """
    Read the configured RAG corpus directory
    and verify that it exists.
    """

    raw_path = global_config.get(
        "rag_folder_path",
        ""
    )

    if (
        not isinstance(raw_path, str)
        or not raw_path.strip()
    ):
        raise HTTPException(
            status_code=400,
            detail="Configure the RAG folder first."
        )

    folder = Path(
        raw_path.strip()
    ).expanduser().resolve()

    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"RAG folder does not exist: {folder}"
        )

    return folder


# -------------------------------------------------
# Prompt processing
# -------------------------------------------------

@app.post("/process_prompt")
def process_prompt(request: PromptRequest):

    if not request.prompt.strip():
        raise HTTPException(400, "Prompt cannot be empty.")
    with app.state.config_lock:
        settings = dict(global_config)
    engine = OllamaEngine()
    engine.set_model(settings["local_llm_model"], "chat", load_tokenizer=False)
    fragments = ""
    used = 0
    try:
        if request.rag_enabled and settings["rag_context_size"]:
            import chromadb
            if not (DB_PATH / "chroma.sqlite3").is_file():
                raise HTTPException(400, "Create the RAG database in settings first.")
            collection = chromadb.PersistentClient(path=str(DB_PATH)).get_collection(
                name=ChromaDBManager.COLLECTION_NAME
            )
            available = collection.count()
            if available:
                embedding_engine = OllamaEngine()
                embedding_engine.set_model(
                    (collection.metadata or {}).get("embedding_model", "qllama/multilingual-e5-small:latest"),
                    "embedding", load_tokenizer=False
                )
                results = collection.query(
                    query_embeddings=[embedding_engine.embed_text(request.prompt, input_type="query")],
                    n_results=min(available, max(5, settings["rag_context_size"] // 100 + 1)),
                    include=["documents"]
                )
                # Leave room for instructions, chat template and the generated answer.
                budget = min(settings["rag_context_size"], max(0,
                    settings["local_llm_context_window"] - len(request.prompt.encode("utf-8")) - 1024))
                fragments, used = bounded_fragments(results["documents"][0], budget)
        system_prompt = PromptPreparator("prompt_templates").prepare_prompt(
            request.prompt, request.rag_enabled, fragments
        )
        response = engine.process_prompt(request.prompt, system_prompt,
                                         settings["local_llm_context_window"])
        if not isinstance(response, str) or not response.strip():
            raise ValueError("The model returned no answer text. Increase the context size or reduce the RAG budget.")
    except HTTPException:
        raise
    except Exception as exc:
        logging.exception("Prompt processing failed")
        raise HTTPException(502, f"Local model or database request failed: {exc}") from exc
    return {"status": "ok", "response": response, "rag_enabled": request.rag_enabled,
            "rag_context_used": used, "rag_budget_unit": "UTF-8 bytes (conservative token budget)"}


class LLMSettingsRequest(BaseModel):
    local_llm_engine: str = "ollama"
    local_llm_model: str = Field(min_length=1)
    local_llm_context_window: int = Field(ge=1024, le=1048576)
    rag_context_size: int = Field(ge=0, le=1048576)


def save_config(updates):
    with app.state.config_lock:
        updated = {**global_config, **updates}
        temp_path = CONFIG_PATH.with_suffix(".json.tmp")
        temp_path.write_text(json.dumps(updated, indent=4, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(CONFIG_PATH)
        global_config.update(updated)


@app.get("/llm_settings")
def get_llm_settings():
    with app.state.config_lock:
        return dict(global_config)


@app.get("/llm_models")
def get_llm_models():
    try:
        return {"models": model_catalog(), "hardware": hardware_info()}
    except Exception as exc:
        raise HTTPException(503, f"Cannot connect to Ollama. Run ollama serve. {exc}") from exc


@app.get("/llm_model_details")
def get_llm_model_details(model: str):
    try:
        return model_details(model)
    except Exception as exc:
        raise HTTPException(503, f"Cannot inspect model: {exc}") from exc


@app.post("/llm_settings")
def set_llm_settings(request: LLMSettingsRequest):
    if request.local_llm_engine != "ollama":
        raise HTTPException(400, "Only Ollama is supported.")
    if request.rag_context_size > request.local_llm_context_window - 1024:
        raise HTTPException(400, "Reserve at least 1024 context tokens outside the RAG budget.")
    try:
        models = model_catalog()
        if request.local_llm_model not in [m["name"] for m in models]:
            raise HTTPException(400, "Select an installed Ollama model.")
        details = model_details(request.local_llm_model)
        if details["capabilities"] and "completion" not in details["capabilities"]:
            raise HTTPException(400, "Select a model supporting text completion.")
        if details["context_length"] and request.local_llm_context_window > details["context_length"]:
            raise HTTPException(400, "Context exceeds the model context length.")
        save_config(request.model_dump() if hasattr(request, "model_dump") else request.dict())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Cannot save model settings: {exc}") from exc
    return get_llm_settings()


# -------------------------------------------------
# RAG folder configuration
# -------------------------------------------------

@app.post("/set_rag_folder_path")
def set_rag_folder_path(request: PathRequest):

    if not request.rag_path.strip():
        raise HTTPException(
            status_code=400,
            detail="The directory path cannot be empty."
        )

    folder = Path(
        request.rag_path.strip()
    ).expanduser().resolve()

    if not folder.is_dir():
        raise HTTPException(
            status_code=400,
            detail=(
                "Your input path does not exist "
                "or is not a directory."
            )
        )

    with app.state.rag_lock:

        if app.state.rag_status["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=(
                    "Cannot change the RAG folder "
                    "during indexing."
                )
            )

        save_config({"rag_folder_path": str(folder)})

        app.state.rag_status = initial_rag_status()

    return {
        "status": "ok",
        "message": "The path has been successfully modified.",
        "rag_folder_path": str(folder),
    }


@app.get("/check_current_rag_folder_path")
def check_current_rag_folder_path():

    return {
        "rag_folder_path": global_config.get(
            "rag_folder_path",
            ""
        )
    }


# -------------------------------------------------
# RAG background worker
# -------------------------------------------------

def _run_rag_build(folder: Path):
    """
    Perform database creation in a background thread.

    The worker updates app.state.rag_status whenever
    ChromaDBManager reports progress.
    """

    def on_progress(progress):

        with app.state.rag_lock:
            app.state.rag_status.update(progress)

    try:

        manager = ChromaDBManager()

        result = manager.create_db(
            collection_path=str(DB_PATH),
            source_corpus_folder=str(folder),
            progress_callback=on_progress,
        )

        with app.state.rag_lock:

            app.state.rag_status.update(result)

            if result["failed_files"]:

                app.state.rag_status["status"] = (
                    "completed_with_errors"
                )

            else:

                app.state.rag_status["status"] = (
                    "completed"
                )

    except Exception as exc:

        logging.exception(
            "RAG indexing failed"
        )

        with app.state.rag_lock:

            app.state.rag_status["status"] = "error"

            app.state.rag_status["error"] = str(exc)


# -------------------------------------------------
# Start RAG database creation/update
# -------------------------------------------------

@app.post(
    "/start_rag_database_creation_update",
    status_code=202
)
def create_or_update_rag_database():
    """
    Start RAG database creation in a worker thread.

    Return immediately so that JavaScript can
    poll the API for progress.
    """

    with app.state.rag_lock:

        # Prevent simultaneous indexing operations.
        if app.state.rag_status["status"] == "running":

            raise HTTPException(
                status_code=409,
                detail="RAG indexing is already running."
            )

        # Read the configured RAG directory.
        folder = current_rag_folder()

        # Count available documents.
        total_files = len(
            CorpusFolderIterator(
                str(folder)
            ).get_document_files()
        )

        if total_files == 0:

            raise HTTPException(
                status_code=400,
                detail=(
                    "No supported documents found "
                    "in the RAG folder."
                )
            )

        # Reset indexing statistics.
        app.state.rag_status = initial_rag_status(
            status="running"
        )

        app.state.rag_status["total_files"] = total_files

        # Start background worker.
        worker = Thread(
            target=_run_rag_build,
            args=(folder,),
            daemon=True
        )

        worker.start()

        return dict(app.state.rag_status)


# -------------------------------------------------
# Count existing indexed documents
# -------------------------------------------------

def _get_indexed_file_count():
    """
    Inspect an existing database without creating
    a new collection.
    """

    if not (DB_PATH / "chroma.sqlite3").is_file():
        return 0

    import chromadb

    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    try:

        collection = client.get_collection(
            name=ChromaDBManager.COLLECTION_NAME
        )

    except Exception:
        return 0

    metadata = collection.get(
        include=["metadatas"]
    )["metadatas"] or []

    return len({
        meta["file_path"]
        for meta in metadata
        if meta and meta.get("file_path")
    })


# -------------------------------------------------
# Get current RAG indexing status
# -------------------------------------------------

@app.get("/get_current_rag_status")
def get_current_rag_status():
    """
    Return live indexing progress.

    If indexing has not started, count documents
    in the configured directory and existing database.
    """

    with app.state.rag_lock:

        snapshot = dict(
            app.state.rag_status
        )

    # Return current progress if indexing has started.
    if snapshot["status"] != "idle":
        return snapshot

    # Count source documents.
    try:

        folder = current_rag_folder()

        snapshot["total_files"] = len(
            CorpusFolderIterator(
                str(folder)
            ).get_document_files()
        )

    except HTTPException:
        pass

    # Count documents already indexed.
    try:

        snapshot["db_size"] = (
            _get_indexed_file_count()
        )

    except Exception:

        logging.exception(
            "Could not inspect existing RAG database"
        )

    return snapshot