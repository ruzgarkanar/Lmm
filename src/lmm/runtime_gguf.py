"""GGUF/int4 runtime — the CHEAP/LOCAL build of LMM's LANGUAGE engine. The thesis
is proven here: "runs fluently on CPU, without an expensive GPU".

Contract is the SAME as `runtime.py`: `generate(messages, max_tokens, temperature,
system)` → plain text. So the rest of `lmm/` can switch to this backend without
any change (LMM_BACKEND=gguf). The model loads once (singleton, lazy, idempotent).

Engine: llama.cpp (llama-cpp-python). Q4_K_M quantized 3B ~2GB; raw transformers
on CPU was ~4.5 tok/s, ~15-25 tok/s expected here.
"""
import os

_LLM = None


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path():
    """GGUF file path. Can be overridden with LMM_GGUF_PATH. Priority:
    the LoRA-merged LMM engine (if present) > raw Qwen quantization."""
    env = os.environ.get("LMM_GGUF_PATH")
    if env:
        return env
    merged = os.path.join(_root(), "models/lmm/lmm-q4_k_m.gguf")
    if os.path.exists(merged):
        return merged
    return os.path.join(_root(), "models/qwen-3b/qwen3b-q4.gguf")


def ready():
    return os.path.exists(path())


def _load():
    global _LLM
    if _LLM is not None:
        return
    if not ready():
        raise FileNotFoundError(
            f"GGUF model not found: {path()} — convert it first "
            "(convert_hf_to_gguf.py + llama-quantize Q4_K_M).")
    from llama_cpp import Llama
    # n_threads=None → llama.cpp picks the physical core count.
    # chat_format: create_chat_completion reads Qwen's own template from the
    # GGUF metadata (embedded during convert), no extra template needed.
    _LLM = Llama(
        model_path=path(),
        n_ctx=int(os.environ.get("LMM_N_CTX", "8192")),
        n_threads=(int(os.environ["LMM_N_THREADS"])
                   if os.environ.get("LMM_N_THREADS") else None),
        n_gpu_layers=int(os.environ.get("LMM_N_GPU_LAYERS", "0")),
        verbose=False,
    )


def device():
    _load()
    n = int(os.environ.get("LMM_N_GPU_LAYERS", "0"))
    return "gpu" if n > 0 else "cpu"


def generate(messages, max_tokens=256, temperature=0.7, system=None):
    """Generates. `messages`: str (a single user utterance) or [{role,content}].
    `system`: the system instruction (for grounding). Returns: plain text.
    Contract identical to runtime.py."""
    _load()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)
    # temperature=0 → greedy (also disable top_p to suppress sampling entirely).
    kwargs = {"messages": list(messages),
              "max_tokens": max_tokens,
              "temperature": temperature}
    if temperature > 0:
        kwargs["top_p"] = 0.9
    else:
        kwargs["top_p"] = 1.0
    resp = _LLM.create_chat_completion(**kwargs)
    return resp["choices"][0]["message"]["content"].strip()
