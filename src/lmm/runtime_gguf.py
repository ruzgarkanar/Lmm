"""GGUF/int4 runtime — the CHEAP/LOCAL build of LMM's LANGUAGE engine. The thesis
is proven here: "runs fluently on CPU, without an expensive GPU".

Contract is the SAME as `runtime.py`: `generate(messages, max_tokens, temperature,
system)` → plain text. So the rest of `lmm/` can switch to this backend without
any change (LMM_BACKEND=gguf). The model loads once (singleton, lazy, idempotent).

Engine: llama.cpp (llama-cpp-python). Q4_K_M quantized 3B ~2GB; raw transformers
on CPU was ~4.5 tok/s, ~15-25 tok/s expected here.
"""
import os

from lmm import paths

_LLM = None


def _root():
    # The user's location, not the package's — see lmm/paths.py for why this
    # is not derived from __file__ any more.
    return paths.home()


def path():
    """GGUF file path. Can be overridden with LMM_GGUF_PATH. Priority:
    the LoRA-merged LMM engine (if present) > raw Qwen quantization."""
    env = os.environ.get("LMM_GGUF_PATH")
    if env:
        return env
    merged = paths.under("models", "lmm", "lmm-q4_k_m.gguf")
    if os.path.exists(merged):
        return merged
    return paths.under("models", "qwen-3b", "qwen3b-q4.gguf")


def ready():
    return os.path.exists(path())


def gpu_layers():
    """How many layers to offload to the accelerator.

    THE DEFAULT USED TO BE 0, and 0 is not a neutral default — it is the CPU
    path, and on this machine the CPU path is measured at 21.7 s per call
    against 0.9 s with Metal. Anyone who did not know to export
    `LMM_N_GPU_LAYERS` ran the local engine ~20x slower than their own hardware
    allows, which on a 130-call ingestion is the difference between a minute
    and three quarters of an hour.

    The detection asks the LIBRARY, not the platform: `llama_supports_gpu_offload`
    is llama.cpp's own report of whether THIS build was compiled with a GPU
    backend (Metal, CUDA, ROCm, Vulkan, SYCL). A CPU-only wheel on a machine
    with a GPU answers False and we stay on the CPU, which is correct — the
    wheel cannot use the card. No platform string is parsed and no device is
    probed by name.

    `-1` is llama.cpp's "all layers"; the loader clamps it to the model's own
    layer count, so it needs no knowledge of the model.

    `LMM_N_GPU_LAYERS` remains an explicit override in BOTH directions: set it
    to 0 to force the CPU path (that is how the measurement below was taken),
    or to a smaller number when the model does not fit in VRAM.
    """
    env = os.environ.get("LMM_N_GPU_LAYERS", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    try:
        from llama_cpp import llama_supports_gpu_offload
        return -1 if llama_supports_gpu_offload() else 0
    except Exception:
        # An older//odd llama_cpp build without the query: the safe answer is
        # the one that always runs.
        return 0


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
        n_gpu_layers=gpu_layers(),
        verbose=False,
    )


def device():
    _load()
    return "gpu" if gpu_layers() != 0 else "cpu"


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
