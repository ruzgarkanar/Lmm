"""Qwen runtime — LMM's LANGUAGE engine. Loads, generates. condition-1 lives here.

Its responsibility is single: 'give text, get text'. It makes no decisions —
Qwen only produces candidates, decision logic lives in `verify`/`session`. The
model loads once (singleton).

Device is automatic: MPS (Apple) · CUDA · else CPU. In deployment the GGUF/int4
build will plug in here (speed on cheap hardware); transformers for now.
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
            f"Qwen model not found: {path()} — download it first "
            "(huggingface: Qwen/Qwen2.5-3B-Instruct).")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _DEVICE = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
    _TOK = AutoTokenizer.from_pretrained(path())
    dtype = torch.float16 if _DEVICE in ("mps", "cuda") else torch.float32
    _MODEL = (AutoModelForCausalLM.from_pretrained(path(), dtype=dtype)
              .to(_DEVICE).eval())
    # Apply the LoRA adapter (if present) — fine-tuned BEHAVIOR
    # (language/identity/consistency). Behavior, not knowledge; the graph still
    # grows without retraining (condition-3). Disable with LMM_NO_LORA=1
    # (back to the raw model — A/B comparison).
    adapter = os.path.join(_root(), "models/lmm/lora")
    if os.path.isdir(adapter) and os.environ.get("LMM_NO_LORA") != "1":
        try:
            from peft import PeftModel
            _MODEL = PeftModel.from_pretrained(_MODEL, adapter).eval()
        except Exception:                                   # noqa: BLE001
            pass        # no peft / broken adapter → continue with the raw model


def device():
    _load()
    return _DEVICE


def generate(messages, max_tokens=256, temperature=0.7, system=None):
    """Generates. `messages`: str (a single user utterance) or [{role,content}].
    `system`: the system instruction (for grounding). Returns: plain text.

    Backend selection: if LMM_BACKEND=gguf, delegates to the llama.cpp/int4
    build (speed on cheap/local hardware). Otherwise transformers. Same contract."""
    if os.environ.get("LMM_BACKEND") == "gguf":
        from . import runtime_gguf
        return runtime_gguf.generate(messages, max_tokens=max_tokens,
                                     temperature=temperature, system=system)
    if os.environ.get("LMM_BACKEND") == "azure":     # EXPERIMENT: same-engine comparison
        from . import runtime_azure
        return runtime_azure.generate(messages, max_tokens=max_tokens,
                                      temperature=temperature, system=system)
    import torch
    _load()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)
    text = _TOK.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True)
    # truncation: an overly long message must not exceed the model's position
    # limit and crash.
    ids = _TOK(text, return_tensors="pt", truncation=True,
               max_length=8192).to(_DEVICE)
    # Sampling parameters are passed only when do_sample=True (in greedy mode
    # transformers printed an "unused flags" warning — behavior correct, noise).
    kwargs = {"max_new_tokens": max_tokens, "pad_token_id": _TOK.eos_token_id}
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs["do_sample"] = False
    with torch.no_grad():
        out = _MODEL.generate(**ids, **kwargs)
    return _TOK.decode(out[0][ids.input_ids.shape[1]:],
                       skip_special_tokens=True).strip()
