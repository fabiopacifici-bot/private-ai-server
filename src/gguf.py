import os
import threading
import time
from typing import List, Dict

import config
import state
import model_cache as mcache

# The shared Llama instance (MAX_CACHE_MODELS=1) is NOT thread-safe: concurrent
# requests on the same instance crashed CUDA (observed on Qwen3-30B-A3B). All
# GGUF inference — streaming and non-streaming — must run under this lock.
infer_lock = threading.Lock()


def build_llama_cpp_prompt(messages: List[Dict[str, str]], template: str = "generic") -> str:
    """Build a llama.cpp chat prompt for the given message list.

    ``template`` selects the chat-markup dialect:
    - ``generic``: ``<|system|>/<|user|>/<|assistant|>`` (default, generic
      llama.cpp ``chat`` style)
    - ``qwen3``: ChatML ``<|im_start|>...<|im_end|>`` with a ``<|im_start|>assistant``
      suffix — required for Qwen3-family GGUFs (Qwen/Qwen3-30B-A3B etc.); the
      legacy generic markup makes them emit only a bare special token.

    Roles are mapped per-dialect; unknown roles (e.g. ``tool``) degrade to
    ``user``. The trailing assistant prompt is appended in the correct dialect.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if template == "qwen3":
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        else:  # generic
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            else:  # user, tool, or any unknown role → treat as user
                parts.append(f"<|user|>\n{content}")
    if template == "qwen3":
        parts.append("<|im_start|>assistant\n")
    else:
        parts.append("<|assistant|>")
    return "\n".join(parts)


def detect_chat_template(model_name: str = "", gguf_meta: dict | None = None) -> str:
    """Return the chat-template dialect for a GGUF model.

    Priority:
    1. An explicit ``chat_template``-style key in ``gguf_meta`` (if present) that
       contains ``<|im_start|>`` → ``qwen3``.
    2. Filename/model-name heuristic: tokens indicating a Qwen3-family model
       (``qwen3``, plus ``moe``/``a3b``/``a17b``/``a18b`` variants) → ``qwen3``.
    3. Fallback: ``generic``.

    ``model_name`` may be a full HF repo id, a file path, or a bare GGUF
    filename; matching is case-insensitive.
    """
    meta = gguf_meta or {}
    for key in ("chat_template", "tokenizer.chat_template", "qwen3.chat_template"):
        val = meta.get(key)
        if isinstance(val, str) and "<|im_start|>" in val:
            return "qwen3"
    name = (model_name or "").lower()
    # Qwen3-family: also catches "qwen3.5", "qwen3-moe", and the official
    # "Qwen3-30B-A3B" / "Qwen3.8-...-A17B" MoE naming.
    if "qwen3" in name and not ("qwen2.5" in name or "qwen2" in name):
        return "qwen3"
    return "generic"


def load_gguf(model_name: str, gguf_path: str) -> None:
    if not config.llama_cpp_available:
        raise RuntimeError("llama-cpp-python is not installed. Install: pip install llama-cpp-python")
    if not os.path.isfile(gguf_path):
        raise RuntimeError(f"GGUF file not found: {gguf_path}")

    n_gpu_layers = int(os.getenv("GGUF_N_GPU_LAYERS", "-1"))  # -1 offloads all layers to GPU
    n_ctx = int(os.getenv("GGUF_N_CTX", "4096"))
    # Prompt-processing batch size. Benchmarked on Qwen3-30B-A3B Q2_K (4090 16GB,
    # this llama-cpp-python build): 512 -> 126, 1024 -> 146, 2048 -> 137 t/s.
    # 1024 is the sweet spot and the default.
    n_batch = int(os.getenv("GGUF_N_BATCH", "1024"))
    config.log.info("Loading GGUF with llama-cpp-python: %s (n_gpu_layers=%s, n_ctx=%s, n_batch=%s)",
                    gguf_path, n_gpu_layers, n_ctx, n_batch)

    start_ns = time.time_ns()
    try:
        mdl = config.LlamaCpp(model_path=gguf_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx,
                              n_batch=n_batch, verbose=False)
    except Exception as e:
        config.log.exception("llama-cpp-python failed for %s", gguf_path)
        msg = str(e)
        # Detect the most common cause: GGUF uses an architecture llama.cpp doesn't know yet
        if "unknown model architecture" in msg:
            arch = msg.split("'")[-2] if "'" in msg else "?"
            raise RuntimeError(
                f"GGUF uses architecture '{arch}' which your installed llama-cpp-python "
                f"does not recognize. Upgrade: pip install -U llama-cpp-python. "
                f"If the model is brand new, you may need a llama.cpp build that supports it "
                f"from source. The Transformers backend cannot load GGUF files directly — "
                f"you need the matching safetensors repo on Hugging Face."
            ) from e
        raise RuntimeError(
            f"llama-cpp-python failed to load GGUF: {type(e).__name__}: {msg}. "
            f"Common causes: (1) llama-cpp-python was built without CUDA/GPU support — "
            f"set GGUF_N_GPU_LAYERS=0 to run on CPU; (2) GGUF file is corrupt or incomplete; "
            f"(3) insufficient RAM for n_ctx={n_ctx} — try GGUF_N_CTX=512."
        ) from e
    mcache.cache_model(model_name, "gguf_llama_cpp", mdl, extra_meta={
        "load_duration_ns": time.time_ns() - start_ns,
        "local_path": gguf_path,
        "n_gpu_layers": n_gpu_layers,
        "n_ctx": n_ctx,
        "n_batch": n_batch,
    })
