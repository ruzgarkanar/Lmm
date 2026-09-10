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


def parallel_ok():
    """Can this backend take concurrent calls? An HTTP endpoint can; a local
    model owns one set of weights and one device queue, and cannot."""
    return os.environ.get("LMM_BACKEND") in ("azure", "openai")


def parallel_map(fn, items, workers=8):
    """Run independent engine calls side by side WHEN THE BACKEND CAN TAKE
    IT. Only the runtime knows: an HTTP endpoint (azure/openai) serves
    concurrent requests as a matter of course; a local model owns one set
    of weights and one device queue, so parallel callers would only fight
    over it — there the map stays serial, byte for byte the old behaviour.
    Order is preserved either way, so every caller stays deterministic.
    Measured need: the verifier's per-sentence re-reads — seven independent
    ~1.5s calls that used to stand in single file (10.4s of a 15.9s
    greeting turn was this queue)."""
    items = list(items)
    if len(items) < 2 or not parallel_ok():
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        return list(pool.map(fn, items))


def generate_stream(messages, max_tokens=256, temperature=0.7, system=None):
    """Like `generate`, but a GENERATOR of text chunks — the engine's words
    as they arrive. Only the HTTP backends truly stream; everywhere else
    the whole reply is yielded once, so a consumer written against this
    contract degrades to the batch behaviour by itself. The gate this
    exists for (the composer's line reader) is pure string arithmetic, so
    a streamed line can be judged and handed on while the engine is still
    writing the next."""
    backend = os.environ.get("LMM_BACKEND")
    if backend == "azure":
        from . import runtime_azure
        yield from runtime_azure.generate_stream(
            messages, max_tokens=max_tokens, temperature=temperature,
            system=system)
        return
    if backend == "openai":
        from . import runtime_openai
        yield from runtime_openai.generate_stream(
            messages, max_tokens=max_tokens, temperature=temperature,
            system=system)
        return
    yield generate(messages, max_tokens=max_tokens, temperature=temperature,
                   system=system)


def small_backend():
    """Which backend answers the turn's SMALL questions — the yes/no
    classifiers and one-word namings (is this an identity question, does
    this ask for material, what language is this). They are the calls a
    cascade sends to a small local model in every production writeup on
    cost, and the one we measured agreed with the hosted engine 19/20
    on fresh multilingual examples.

    OFF unless the operator sets `LMM_SMALL_BACKEND` (same values as
    LMM_BACKEND; "local" for the transformers default): only
    wants_material has a measured agreement, and a switch shipped on one
    measurement ships OFF. Returns the backend name or None."""
    small = os.environ.get("LMM_SMALL_BACKEND", "").strip()
    if not small:
        return None
    return None if small == os.environ.get("LMM_BACKEND") else small


def generate(messages, max_tokens=256, temperature=0.7, system=None,
             small=False):
    # THE SMALL ROAD IS A NAME, NOT AN ENVIRONMENT MUTATION: half this
    # codebase runs its calls side by side (`parallel_map`), and a
    # thread that edits LMM_BACKEND edits every thread's engine.
    chosen = small_backend() if small else None
    return _dispatch(chosen, messages, max_tokens, temperature, system)


def _dispatch(backend_name, messages, max_tokens=256, temperature=0.7,
              system=None):
    """Generates. `messages`: str (a single user utterance) or [{role,content}].
    `system`: the system instruction (for grounding). Returns: plain text.

    Backend selection, by LMM_BACKEND — every one of them keeps this exact
    contract, because the graph and the gate never learn which one is running:
    `gguf` is the llama.cpp/int4 build (speed on cheap local hardware),
    `openai` is any OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, a
    vLLM or Ollama server — one client and a base_url), `azure` is Azure's own
    client, which needs a different one because a deployment name and an
    api-version are a shape only Azure has. Unset means the local transformers
    model, which is the product default."""
    backend = backend_name or os.environ.get("LMM_BACKEND")
    if backend == "local":
        backend = None                  # the transformers default, by name
    if backend == "gguf":
        from . import runtime_gguf
        return runtime_gguf.generate(messages, max_tokens=max_tokens,
                                     temperature=temperature, system=system)
    if backend == "openai":
        from . import runtime_openai
        return runtime_openai.generate(messages, max_tokens=max_tokens,
                                       temperature=temperature, system=system)
    if backend == "azure":     # EXPERIMENT: same-engine comparison
        from . import runtime_azure
        return runtime_azure.generate(messages, max_tokens=max_tokens,
                                      temperature=temperature, system=system)
    # A MISSING ENGINE SAYS WHICH ENGINE IS MISSING. The package installs
    # with no dependencies — that is the design claim — so a first run on
    # a fresh machine reaches the default local engine and, until this,
    # told the reader "No module named 'torch'". Nothing in that sentence
    # says an engine was needed, that there are four to choose from, or
    # that the cheapest is one environment variable away. The reader is
    # not missing torch; the reader is missing an ENGINE. Same manner as
    # the optional readers ("lmm[pdf]"), applied to the one import that
    # every first run hits.
    try:
        import torch                                        # noqa: F401
    except ImportError as missing:
        # THE NAME ON PyPI IS NOT THE NAME IN THE IMPORT, and it is written
        # once — the optional readers already keep it, and a name written
        # in two places is eventually two different names. Telling a
        # reader to `pip install lmm[local]` would send them to somebody
        # else's project.
        from .tables import DIST as dist
        raise ImportError(
            "LMM has no engine to speak with. Choose one:\n"
            "  LMM_BACKEND=openai  — any OpenAI-compatible endpoint "
            "(OpenAI, OpenRouter, Groq, vLLM, Ollama): set OPENAI_API_KEY "
            "and OPENAI_BASE_URL\n"
            "  LMM_BACKEND=azure   — Azure OpenAI: set "
            "AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT\n"
            f"  LMM_BACKEND=gguf    — llama.cpp on your own machine: "
            f"pip install '{dist}[gguf]', set LMM_GGUF to the model file\n"
            f"  (unset)             — a local transformers model: "
            f"pip install '{dist}[local]'\n"
            "The graph half needs none of these: learn(..., deep=False) "
            "and where() run with no engine at all."
        ) from missing
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
