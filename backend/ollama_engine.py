from pathlib import Path

import ollama
from transformers import AutoTokenizer


class OllamaEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_type = None  # "chat" or "embedding"

    def set_model(self, model: str, model_type: str = None, load_tokenizer=True):
        self.model = model

        # Infer model type if not explicitly provided
        if model_type is None:
            if "e5" in model.lower() or "embed" in model.lower():
                model_type = "embedding"
            else:
                model_type = "chat"

        self.model_type = model_type
        self.tokenizer = None

        # Tokenizer for models where we know the HF equivalent
        if load_tokenizer and model == "qwen3.5:4b":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3.5-4B"
            )

        elif load_tokenizer and "multilingual-e5-small" in model.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
                "intfloat/multilingual-e5-small"
            )

    def check_models(self):
        return ollama.list()

    def process_prompt(self, user_prompt: str, system_prompt: str = "", context_size=4096):
        if self.model is None:
            raise ValueError("Ollama model is not set.")

        if self.model_type == "embedding":
            raise ValueError(
                f"Model '{self.model}' is configured as an embedding model "
                "and cannot be used for chat."
            )

        # Qwen 3/3.5 supports disabling thinking. With small RAG contexts,
        # thinking can consume generation space before any final answer appears.
        chat_options = {"think": False} if self.model.split("/")[-1].lower().startswith("qwen3") else {}
        resp = ollama.chat(
            **chat_options,
            stream=False,
            model=self.model,
            keep_alive=60,
            options={"num_ctx": context_size},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        content = resp["message"].get("content", "")
        if not isinstance(content, str) or not content.strip():
            reason = resp.get("done_reason") or "unknown"
            thinking_only = bool(resp["message"].get("thinking"))
            raise ValueError(
                f"Ollama returned no answer text (stop reason: {reason}, "
                f"thinking-only: {thinking_only}). Increase the model context size "
                "in LLM settings or reduce the RAG budget."
            )
        return content

    def tokenize_input(self, input_text: str):
        if self.tokenizer is None:
            raise ValueError(
                f"Tokenizer is not initialized for model '{self.model}'."
            )

        return self.tokenizer.encode(
            input_text,
            add_special_tokens=False
        )

    def get_token_count(self, input_text: str):
        return len(self.tokenize_input(input_text))

    def assemble_tokens_into_text(self, tokens: list):
        if self.tokenizer is None:
            raise ValueError(
                f"Tokenizer is not initialized for model '{self.model}'."
            )

        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=True
        )

    def embed_text(
        self,
        text: str,
        input_type: str = None
    ) -> list[float]:
        """
        Generate an embedding using Ollama.

        input_type:
            None       -> use text unchanged
            "query"    -> prepend 'query: ' for E5 models
            "passage"  -> prepend 'passage: ' for E5 models
        """

        if self.model is None:
            raise ValueError(
                "Ollama model is not set. "
                "Please set the model before embedding text."
            )

        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        if self.model_type != "embedding":
            raise ValueError(
                f"Model '{self.model}' is not configured as an embedding model."
            )

        embedding_input = text.strip()

        # E5-specific formatting
        if "e5" in self.model.lower():
            if input_type == "query":
                embedding_input = f"query: {embedding_input}"

            elif input_type == "passage":
                embedding_input = f"passage: {embedding_input}"

            elif input_type is not None:
                raise ValueError(
                    "input_type must be 'query', 'passage', or None."
                )

        resp = ollama.embed(
            model=self.model,
            input=embedding_input
        )

        embeddings = resp.embeddings

        if not embeddings:
            raise ValueError(
                f"Ollama returned no embeddings for model '{self.model}'."
            )

        embedding = embeddings[0]

        if not embedding:
            raise ValueError("Returned embedding is empty.")

        return embedding




class PromptPreparator:

    def __init__(self, prompt_templates_path):
        self.prompt = None

        backend_dir = Path(__file__).resolve().parent

        self.prompt_templates_path = backend_dir / prompt_templates_path

        print("Prompt templates directory:", self.prompt_templates_path)

    def prepare_prompt(self, prompt, rag_needed=False, fragments=""):
        basic = (self.prompt_templates_path / "basic_system_prompt.txt").read_text(
            encoding="utf-8"
        )
        if not rag_needed:
            return basic
        instructions = (self.prompt_templates_path / "rag_system_prompt.txt").read_text(
            encoding="utf-8"
        )
        return basic + "\n" + instructions + "\n<rag_fragments>\n" + fragments + "\n</rag_fragments>"
