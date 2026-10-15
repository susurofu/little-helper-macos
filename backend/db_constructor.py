"""Create and incrementally synchronize a local Chroma RAG collection."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb

from backend.corpus_parser import CorpusFolderIterator, CorpusParser
from backend.ollama_engine import OllamaEngine


class ChromaDBManager:
    COLLECTION_NAME = "Database"

    def __init__(self):
        self.collection_path = None
        self.source_corpus_folder = None
        self.client = None
        self.collection = None

        self.ollama_model = "qwen3.5:4b"
        self.ollama_engine = OllamaEngine()
        self.ollama_engine.set_model(
            self.ollama_model,
            model_type="chat"
        )

        self.embedding_model = "qllama/multilingual-e5-small:latest"
        self.embedding_engine = OllamaEngine()
        self.embedding_engine.set_model(
            self.embedding_model,
            model_type="embedding"
        )

        self.chunk_size = 300

    def set_ollama_model(self, model: str):
        self.ollama_model = model
        self.ollama_engine.set_model(
            model,
            model_type="chat"
        )

    def set_chunck_size(self, chunk_size: int):
        # Keep the existing method name for compatibility.
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.chunk_size = chunk_size

    def initialize_custom_embedding_model(self, model: str):
        self.embedding_model = model
        self.embedding_engine = OllamaEngine()
        self.embedding_engine.set_model(
            model,
            model_type="embedding"
        )

    def _check_collection_compatibility(self):
        """An existing collection must not mix incompatible embeddings."""

        metadata = self.collection.metadata or {}
        old_model = metadata.get("embedding_model")
        if old_model and old_model != self.embedding_model:
            raise ValueError(
                f"Collection uses embedding model {old_model!r}, "
                f"but the current model is {self.embedding_model!r}. "
                "Use the original model or create a separate database."
            )

        old_chunk_size = metadata.get("chunk_size")
        if (old_chunk_size is not None and int(old_chunk_size) != self.chunk_size
        ):
            raise ValueError(
                f"Collection uses chunk_size={old_chunk_size}, "
                f"but current chunk_size={self.chunk_size}. "
                "Use a separate database to change chunk size."
            )

    def initialize_existing_db(self, collection_path: str):
        db_path = Path(collection_path).expanduser().resolve()
        if not (db_path / "chroma.sqlite3").is_file():
            raise FileNotFoundError(
                f"Chroma database does not exist: {db_path}"
            )

        self.collection_path = str(db_path)
        self.client = chromadb.PersistentClient(path=str(db_path))

        self.collection = self.client.get_collection(name=self.COLLECTION_NAME)

        self._check_collection_compatibility()

    def create_db(self, collection_path: str, source_corpus_folder: str, progress_callback=None):
        """
        Synchronize indexed files with a source folder.

        The optional callback receives a dictionary containing
        progress counts after each file, including unchanged files.

        The returned dictionary has the same structure.
        """

        if not collection_path or not source_corpus_folder:
            raise ValueError("Both the database path and source folder are required")

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        db_path = Path(collection_path).expanduser().resolve()
        folder = Path(source_corpus_folder).expanduser().resolve()

        if not folder.is_dir():
            raise FileNotFoundError(f"Source corpus folder is missing: {folder}")

        files = sorted(
            CorpusFolderIterator(
                str(folder)
            ).get_document_files()
        )

        # Reject empty folders to avoid accidentally deleting
        # an existing index.
        if not files:
            raise ValueError(f"No supported documents found in {folder}")

        self.collection_path = str(db_path)
        self.source_corpus_folder = str(folder)

        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(db_path))

        # Check compatibility before writing collection metadata.
        names = {
            item if isinstance(item, str) else item.name
            for item in self.client.list_collections()
        }

        if self.COLLECTION_NAME in names:

            self.collection = self.client.get_collection(name=self.COLLECTION_NAME)
            self._check_collection_compatibility()

        else:
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_model": self.embedding_model,
                    "chunk_size": self.chunk_size,
                },
            )

        # Read existing metadata.
        existing = self.collection.get(include=["metadatas"])
        existing_by_path = {}
        for item_id, meta in zip(
            existing["ids"],
            existing["metadatas"] or []
        ):

            if meta and meta.get("file_path"):

                existing_by_path.setdefault(
                    meta["file_path"],
                    []
                ).append(
                    (item_id, meta)
                )

        indexed_paths = set(existing_by_path)

        current_paths = {
            str(file.resolve())
            for file in files
        }

        parser = CorpusParser(files)

        stats = {
            "total_files": len(files),
            "processed_files": 0,
            "db_size": len(indexed_paths),
            "updated_files": 0,
            "unchanged_files": 0,
            "empty_files": 0,
            "failed_files": 0,
            "removed_files": 0,
            "current_file": None,
            "last_file_error": None,
        }

        def publish():

            stats["db_size"] = len(indexed_paths)
            if progress_callback is not None:
                progress_callback(dict(stats))

        publish()
        # Process documents.
        for file in files:

            file_path = str(file.resolve())
            stats["current_file"] = file_path
            previous = existing_by_path.get(file_path,[])

            try:
                stat = file.stat()
                # Determine whether the document was modified.
                unchanged = bool(previous) and all(
                    "date_modified" in meta
                    and abs(
                        float(meta["date_modified"]) - stat.st_mtime
                    ) < 1e-6
                    and (
                        "file_size" not in meta
                        or int(meta["file_size"]) == stat.st_size
                    )
                    and "total_chunks" in meta
                    and len(previous) == int(meta["total_chunks"])
                    for _, meta in previous
                )

                if unchanged:
                    stats["unchanged_files"] += 1
                    continue

                # Parse the document.
                text = parser.parse_file(file)

                if text is None:
                    raise ValueError(
                        "No text extracted "
                        "(possibly unreadable document)"
                    )

                if not text.strip():
                    # Remove previously indexed chunks if the
                    # document is now empty.
                    if previous:
                        self.collection.delete(
                            ids=[
                                item_id
                                for item_id, _ in previous
                            ]
                        )

                        indexed_paths.discard(file_path)
                    stats["empty_files"] += 1
                    continue

                # Tokenize the document.
                tokens = self.ollama_engine.tokenize_input(text)

                chunks, token_chunks = self.chunk_text(
                    tokens,
                    self.chunk_size
                )

                # Generate a stable document identifier.
                fingerprint = hashlib.sha256(
                    file_path.encode("utf-8")
                ).hexdigest()

                # Generate a version identifier.
                version = hashlib.sha256(
                    f"{stat.st_mtime_ns}:{stat.st_size}".encode(
                        "utf-8"
                    )
                ).hexdigest()[:16]

                created_at = datetime.now(timezone.utc).timestamp()

                ids = []
                documents = []
                metadatas = []
                embeddings = []

                # Generate all embeddings before deleting old chunks.
                for i, (chunk, chunk_tokens) in enumerate(
                    zip(chunks, token_chunks)
                ):

                    if not chunk.strip():
                        continue

                    ids.append(
                        f"chunk:{fingerprint}:{version}:{i}"
                    )

                    documents.append(chunk)
                    metadatas.append({
                        "file_path": file_path,
                        "full_doc_id": f"doc:{fingerprint}",
                        "token_size": len(chunk_tokens),
                        "full_file_token_size": len(tokens),
                        "file_size": stat.st_size,
                        "date_modified": stat.st_mtime,
                        "date_added": created_at,
                    })

                    embedding = self.embedding_engine.embed_text(
                        chunk,
                        input_type="passage"
                    )

                    embeddings.append(embedding)

                if not ids:

                    if previous:

                        self.collection.delete(
                            ids=[
                                item_id
                                for item_id, _ in previous
                            ]
                        )

                        indexed_paths.discard(file_path)

                    stats["empty_files"] += 1

                    continue

                # Record expected chunk count for each document.
                for metadata in metadatas:

                    metadata["total_chunks"] = len(ids)

                old_ids = {
                    item_id
                    for item_id, _ in previous
                }

                # Insert new chunks in batches.
                try:

                    for offset in range(0, len(ids), 64):

                        end = offset + 64

                        self._insert_entry_into_db(
                            ids[offset:end],
                            documents[offset:end],
                            metadatas[offset:end],
                            embeddings[offset:end]
                        )

                except Exception:

                    # Roll back newly staged chunks.
                    new_ids = [
                        item_id
                        for item_id in ids
                        if item_id not in old_ids
                    ]

                    if new_ids:
                        self.collection.delete(
                            ids=new_ids
                        )

                    raise

                # Remove obsolete chunks after successful insertion.
                obsolete_ids = list(
                    old_ids - set(ids)
                )

                if obsolete_ids:

                    self.collection.delete(
                        ids=obsolete_ids
                    )

                indexed_paths.add(file_path)

                stats["updated_files"] += 1

            except Exception as exc:

                stats["failed_files"] += 1

                stats["last_file_error"] = (
                    f"{file.name}: {exc}"
                )

                # Check whether the document is still indexed.
                try:

                    remaining = self.collection.get(
                        where={
                            "file_path": file_path
                        },
                        include=[]
                    )["ids"]

                    if remaining:
                        indexed_paths.add(file_path)
                    else:
                        indexed_paths.discard(file_path)

                except Exception:
                    pass

            finally:

                stats["processed_files"] += 1

                publish()

        # Remove documents that no longer exist in the folder.
        for stale_path in set(existing_by_path) - current_paths:

            self.collection.delete(
                ids=[
                    item_id
                    for item_id, _ in existing_by_path[stale_path]
                ]
            )

            indexed_paths.discard(stale_path)

            stats["removed_files"] += 1

        stats["current_file"] = None

        publish()

        return dict(stats)

    def _insert_entry_into_db(
        self,
        ids,
        documents,
        metadatas,
        embeddings=None
    ):

        if self.collection is None:
            raise ValueError("Database collection is not initialized")

        if embeddings is None:

            embeddings = [
                self.embedding_engine.embed_text(
                    doc,
                    input_type="passage"
                )
                for doc in documents
            ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def get_db_size(
        self,
        unique_files=True,
        chunks=False
    ):
        """
        Count indexed files (default), or stored chunks
        if chunks=True.
        """

        if self.collection is None:
            raise LookupError(
                "Initialize or create the database first"
            )

        if chunks or not unique_files:
            return self.collection.count()

        results = self.collection.get(
            include=["metadatas"]
        )

        return len({
            meta["file_path"]
            for meta in (results["metadatas"] or [])
            if meta and meta.get("file_path")
        })

    def get_query_from_db(
        self,
        query: str,
        n_results: int = 5,
        custom_embeddings=True
    ):
        """
        Query chunks and return a pandas DataFrame.
        """

        import pandas as pd

        if self.collection is None:
            raise ValueError(
                "Database collection is not initialized"
            )

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if n_results < 1:
            raise ValueError(
                "n_results must be positive"
            )

        available = self.collection.count()

        if available == 0:

            return pd.DataFrame(
                columns=[
                    "id",
                    "document",
                    "distance",
                    "metadata"
                ]
            )

        n_results = min(
            n_results,
            available
        )

        if custom_embeddings:

            embedding = self.embedding_engine.embed_text(
                query,
                input_type="query"
            )

            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results
            )

        else:

            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )

        return pd.DataFrame([
            {
                "id": item_id,
                "document": (
                    results.get("documents") or [[]]
                )[0][i],
                "distance": (
                    results.get("distances") or [[]]
                )[0][i],
                "metadata": (
                    results.get("metadatas") or [[]]
                )[0][i],
            }

            for i, item_id in enumerate(
                results["ids"][0]
            )
        ])

    def access_db(self):
        raise NotImplementedError(
            "Use initialize_existing_db()"
        )

    def remove_db(self):
        raise NotImplementedError(
            "Database removal is not implemented"
        )

    def chunk_text(
        self,
        tokenized_text,
        chunk_size=300
    ):
        """Split tokenized text into chunks."""

        if chunk_size <= 0:
            raise ValueError(
                "chunk_size must be positive"
            )

        chunks_detokenized = []
        chunks_tokenized = []

        for i in range(
            0,
            len(tokenized_text),
            chunk_size
        ):

            chunk = tokenized_text[
                i:i + chunk_size
            ]

            chunks_detokenized.append(
                self.ollama_engine.assemble_tokens_into_text(
                    chunk
                )
            )

            chunks_tokenized.append(chunk)

        return (
            chunks_detokenized,
            chunks_tokenized
        )


def assign_unique_id(
    assigned_ids,
    index="doc"
):
    """Kept for backwards compatibility."""

    counter = len(assigned_ids) + 1

    while f"{index}{counter}" in assigned_ids:
        counter += 1

    new_id = f"{index}{counter}"

    assigned_ids.add(new_id)

    return new_id