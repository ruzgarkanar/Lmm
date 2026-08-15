"""Qwen runtime — LMM'in DİL motoru. Yükler, üretir. condition-1 burada yaşar.

Sorumluluğu tek: 'metin ver, metin al'. Hiçbir karar vermez — Qwen yalnız aday
üretir, karar mantığı `verify`/`session`'dadır. Model bir kez yüklenir (tekil).

Aygıt otomatik: MPS (Apple) · CUDA · yoksa CPU. Dağıtımda GGUF/int4 sürümü
buraya takılacak (ucuz donanımda hız); şimdilik transformers.
"""
import os

_MODEL = None
_TOK = None
_DEVICE = None


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path():
    return os.path.join(_root(), "models/qwen-3b")


def ready():
    return os.path.exists(os.path.join(path(), "config.json"))


def _load():
    global _MODEL, _TOK, _DEVICE
    if _MODEL is not None:
        return
    if not ready():
        raise FileNotFoundError(
            f"Qwen modeli bulunamadı: {path()} — önce indir "
            "(huggingface: Qwen/Qwen2.5-3B-Instruct).")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _DEVICE = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
    _TOK = AutoTokenizer.from_pretrained(path())
    dtype = torch.float16 if _DEVICE in ("mps", "cuda") else torch.float32
    _MODEL = (AutoModelForCausalLM.from_pretrained(path(), dtype=dtype)
              .to(_DEVICE).eval())


def device():
    _load()
    return _DEVICE


def generate(messages, max_tokens=256, temperature=0.7, system=None):
    """Üretir. `messages`: str (tek kullanıcı sözü) ya da [{role,content}].
    `system`: sistem talimatı (topraklama için). Dönen: düz metin."""
    import torch
    _load()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)
    text = _TOK.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    # truncation: çok uzun mesaj model pozisyon sınırını aşıp çökmesin.
    ids = _TOK(text, return_tensors="pt", truncation=True,
               max_length=8192).to(_DEVICE)
    # Örnekleme parametreleri yalnız do_sample=True iken geçilir (greedy'de
    # transformers "unused flags" uyarısı basıyordu — davranış doğru, gürültü).
    kwargs = {"max_new_tokens": max_tokens, "pad_token_id": _TOK.eos_token_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs["do_sample"] = False
    with torch.no_grad():
        out = _MODEL.generate(**ids, **kwargs)
    return _TOK.decode(out[0][ids.input_ids.shape[1]:],
                       skip_special_tokens=True).strip()
