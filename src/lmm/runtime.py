"""Qwen runtime — LMM's LANGUAGE engine. Loads, generates. condition-1 lives here.

Its responsibility is single: 'give text, get text'. It makes no decisions —
Qwen only produces candidates, decision logic lives in `verify`/`session`. The
model loads once (singleton).

Device is automatic: MPS (Apple) · CUDA · else CPU. In deployment the GGUF/int4
build will plug in here (speed on cheap hardware); transformers for now.
"""
import os
import time

from lmm import paths

# HOW LONG ONE ENGINE CALL MAY TAKE, in seconds. A measured question took 613
# seconds — not computing, WAITING: the backend was over its per-minute quota
# and the client's retry policy is happy to wait for as long as the quota takes.
# A rate limit is a wait rather than a failure, which is why the retries are
# there and why they stay; what was missing is that a wait has to END.
#
# The default is a budget, not a tuned threshold: it is the point past which a
# person asking a question has stopped waiting for an answer, and it is stated
# here so that anyone who disagrees can set LMM_TIMEOUT — including to 0, which
# means the old behaviour, wait as long as it takes.
DEFAULT_BUDGET = 90.0

_MODEL = None
_TOK = None
_DEVICE = None


def _root():
    # The user's location, not the package's — see lmm/paths.py for why this
    # is not derived from __file__ any more.
    return paths.home()


class EngineTimeout(TimeoutError):
    """One engine call ran past the time budget. It says how long it waited and
    what it was waiting on, because the user's next decision (wait longer, ask
    a smaller question, use another backend) depends on which of those it was."""


def budget():
    """The per-call time budget in seconds; 0 means no limit."""
    raw = (os.environ.get("LMM_TIMEOUT") or "").strip()
    try:
        value = float(raw) if raw else DEFAULT_BUDGET
    except ValueError:
        value = DEFAULT_BUDGET
    return value if value > 0 else 0.0


def within_budget(call, retryable=(Exception,), what="the engine"):
    """Run ONE engine call under the budget, retrying only what is a WAIT.

    `call(remaining)` is handed the seconds it has left, so a backend that can
    take a timeout passes it down instead of being interrupted from outside.
    Anything in `retryable` is treated as a queue rather than a failure and
    tried again with a doubling pause — but only while there is budget left,
    which is the whole point: the previous policy retried a rate limit ten
    times and turned one question into a ten-minute wait with no way out.
    Everything else is raised immediately and unchanged.
    """
    limit = budget()
    if not limit:
        return call(None)
    deadline = time.monotonic() + limit
    pause, last = 1.0, None
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise EngineTimeout(
                f"{what} did not answer within {limit:g}s "
                f"(LMM_TIMEOUT sets this; 0 removes the limit)"
                + (f" — last: {type(last).__name__}: {last}" if last else ""))
        try:
            return call(left)
        except retryable as waiting:
            last = waiting
            left = deadline - time.monotonic()
            if left <= 0:
                continue                # round the loop once to raise, not sleep
            time.sleep(min(pause, left))
            pause *= 2


def path():
    return os.environ.get("LMM_MODEL_PATH") or paths.under("models", "qwen-3b")


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
    adapter = os.environ.get("LMM_LORA_PATH") or paths.under("models", "lmm", "lora")
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
