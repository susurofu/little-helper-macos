"""Local model metadata, approximate memory sizing and bounded RAG text."""
import os
import platform
import subprocess

import ollama


def as_dict(value):
    return value.model_dump() if hasattr(value, "model_dump") else dict(value)


def model_catalog():
    return [{"name": m.get("model") or m.get("name"), "size": m.get("size", 0),
             "details": m.get("details", {})}
            for m in as_dict(ollama.list()).get("models", [])]


def model_details(model):
    data = as_dict(ollama.show(model))
    info = data.get("modelinfo") or data.get("model_info") or {}
    arch = info.get("general.architecture", "")
    def field(name):
        return info.get(f"{arch}.{name}")
    def positive_number(value):
        # Hybrid models may expose per-layer arrays rather than scalar sizes.
        # Those need architecture-specific cache sizing; do not guess a total.
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 else None

    layers = positive_number(field("block_count"))
    heads = positive_number(field("attention.head_count"))
    raw_kv_heads = field("attention.head_count_kv")
    kv_heads = positive_number(raw_kv_heads if raw_kv_heads is not None else heads)
    dimension = positive_number(field("embedding_length"))
    raw_key = field("attention.key_length")
    key = positive_number(raw_key) if raw_key is not None else (
        dimension / heads if dimension and heads else None
    )
    raw_value = field("attention.value_length")
    value = positive_number(raw_value) if raw_value is not None else key
    # FP16 KV cache for scalar transformer metadata. Hybrid/recurrent models
    # return an unavailable estimate instead of failing model inspection.
    kv_bytes = int(layers * kv_heads * (key + value) * 2) if all(
        [layers, kv_heads, key, value]) else None
    return {"context_length": field("context_length"), "kv_bytes_per_token": kv_bytes,
            "capabilities": data.get("capabilities", [])}


def hardware_info():
    ram = None
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        try:
            ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=3))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    unified = platform.system() == "Darwin" and platform.machine() == "arm64"
    gpus = []
    if not unified:
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                timeout=3, stderr=subprocess.DEVNULL, text=True)
            for line in output.splitlines():
                name, memory = line.rsplit(",", 1)
                gpus.append({"name": name.strip(), "memory": int(memory.strip()) * 1024**2})
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {"ram": ram, "unified_memory": unified, "gpus": gpus,
            "note": "GPU detection supports Apple Silicon unified memory and NVIDIA; other GPU memory may be unavailable."}


def bounded_fragments(documents, budget):
    """UTF-8 bytes conservatively bound normal byte-based tokenizer content.

    Include separators in the budget and never split a Unicode character.
    Special-token strings are reference text, not tokenizer control tokens.
    """
    result = ""
    for document in documents:
        if not document:
            continue
        separator = "\n\n" if result else ""
        remaining = budget - len(result.encode("utf-8")) - len(separator.encode("utf-8"))
        if remaining <= 0:
            break
        fragment = document.encode("utf-8")[:remaining].decode("utf-8", errors="ignore")
        if fragment:
            result += separator + fragment
    return result, len(result.encode("utf-8"))
