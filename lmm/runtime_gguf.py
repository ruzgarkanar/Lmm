"""GGUF/int4 runtime — LMM'in DİL motorunun UCUZ/YEREL sürümü. Tez burada kanıtlanır:
"pahalı GPU olmadan, CPU'da akıcı çalışır".

Sözleşme `runtime.py` ile AYNI: `generate(messages, max_tokens, temperature, system)`
→ düz metin. Böylece `lmm/`'in geri kalanı hiç değişmeden bu backend'e geçebilir
(LMM_BACKEND=gguf). Model bir kez yüklenir (tekil, lazy, idempotent).

Motor: llama.cpp (llama-cpp-python). Q4_K_M kuantize 3B ~2GB; transformers ham
CPU'da ~4.5 tok/s idi, buradan ~15-25 tok/s beklenir.
"""
import os

_LLM = None


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path():
    """GGUF dosya yolu. LMM_GGUF_PATH ile override edilebilir. Öncelik:
    LoRA-birleşik LMM motoru (varsa) > ham Qwen kuantizesi."""
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
            f"GGUF modeli bulunamadı: {path()} — önce dönüştür "
            "(convert_hf_to_gguf.py + llama-quantize Q4_K_M).")
    from llama_cpp import Llama
    # n_threads=None → llama.cpp fiziksel çekirdek sayısını seçer.
    # chat_format: create_chat_completion Qwen'in kendi şablonunu GGUF
    # metadata'sından okur (convert sırasında gömülür), ekstra şablon gerekmez.
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
    """Üretir. `messages`: str (tek kullanıcı sözü) ya da [{role,content}].
    `system`: sistem talimatı (topraklama için). Dönen: düz metin.
    Sözleşme runtime.py ile birebir aynı."""
    _load()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)
    # temperature=0 → greedy (top_p'yi de kapatıp örneklemeyi tamamen bastır).
    kwargs = {"messages": list(messages),
              "max_tokens": max_tokens,
              "temperature": temperature}
    if temperature > 0:
        kwargs["top_p"] = 0.9
    else:
        kwargs["top_p"] = 1.0
    resp = _LLM.create_chat_completion(**kwargs)
    return resp["choices"][0]["message"]["content"].strip()
