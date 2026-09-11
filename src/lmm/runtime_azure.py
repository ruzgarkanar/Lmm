"""Azure OpenAI backend — for EXPERIMENTS (same-engine comparison: LMM+4o-mini vs
RAG+4o-mini → isolates the architecture's contribution). NOT the product default:
the product is the local engine (condition-1, cheap hardware); this backend is
benchmark/development only.

Contract is the SAME as runtime.generate: generate(messages, max_tokens,
temperature, system) → plain text. Selection: LMM_BACKEND=azure. Keys from .env
(outside git).
"""
import os

from lmm import paths, runtime
import threading

_CLIENT = None
_LOCK = threading.Lock()    # prevents double client init during parallel ingestion


def _root():
    # The user's location, not the package's — see lmm/paths.py for why this
    # is not derived from __file__ any more.
    return paths.home()


def _env():
    path = paths.find_env()
    if path:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


def _load():
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:     # double-checked: calls are thread-safe (httpx)
                _env()
                from openai import AzureOpenAI
                _CLIENT = AzureOpenAI(
                    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                    api_key=os.environ["AZURE_OPENAI_API_KEY"],
                    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                    # THE RETRIES LIVE IN `runtime.within_budget` NOW. This
                    # endpoint's per-minute quota makes a 429 a WAIT rather
                    # than a failure, and the SDK's default of two retries let
                    # one lose a whole benchmark sample — but retrying ten
                    # times with no clock is how a single measured question
                    # took 613 seconds. Retry as long as there is budget, and
                    # not one round longer; the SDK's own counter is off so the
                    # two policies cannot multiply.
                    max_retries=0)
    return _CLIENT


def _retryable():
    """The errors that mean 'not yet' rather than 'no' — a full queue, a
    dropped connection, the server's own hiccup."""
    from openai import (APIConnectionError, APITimeoutError,   # noqa: PLC0415
                        InternalServerError, RateLimitError)
    return (RateLimitError, APITimeoutError, APIConnectionError,
            InternalServerError)


# WHAT THE DEPLOYMENT DECLARED IT CANNOT TAKE. Newer models refuse
# `max_tokens` (they want `max_completion_tokens`) and refuse any
# temperature but their default; the endpoint SAYS so in its error, and
# that declaration — not a model-name list — is what these flags cache.
# Per-process, learned on the first refusal, never guessed.
_CAPS = {"completion_tokens": False, "no_temperature": False}


def _kwargs(max_tokens, temperature):
    out = {}
    if _CAPS["completion_tokens"]:
        out["max_completion_tokens"] = max_tokens
    else:
        out["max_tokens"] = max_tokens
    if not _CAPS["no_temperature"]:
        out["temperature"] = temperature if temperature > 0 else 0
    return out


def _adapted(said):
    """Read the endpoint's own declaration out of a BadRequest; True if
    a capability flag moved and the call is worth one retry."""
    text = str(said)
    moved = False
    if "max_tokens" in text and "max_completion_tokens" in text \
            and not _CAPS["completion_tokens"]:
        _CAPS["completion_tokens"] = True
        moved = True
    if "temperature" in text and "not support" in text \
            and not _CAPS["no_temperature"]:
        _CAPS["no_temperature"] = True
        moved = True
    return moved


def generate(messages, max_tokens=256, temperature=0.7, system=None):
    client = _load()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)

    def call(left):
        # The remaining budget is handed to the SDK as the request timeout, so
        # a stalled socket is bounded by the same clock as a quota wait.
        api = client if left is None else client.with_options(timeout=left)
        for _try in (1, 2, 3):
            try:
                out = api.chat.completions.create(
                    model=os.environ.get("AZURE_OPENAI_DEPLOYMENT",
                                         "gpt-4o-mini"),
                    messages=messages,
                    **_kwargs(max_tokens, temperature))
                return (out.choices[0].message.content or "").strip()
            except Exception as said:                    # noqa: BLE001
                if not _adapted(said):
                    raise
        raise RuntimeError("the deployment kept refusing its own advice")

    return runtime.within_budget(call, _retryable(), what="the Azure endpoint")


def generate_stream(messages, max_tokens=256, temperature=0.7, system=None):
    """The streamed twin of `generate` — same client, same budget discipline
    on OPENING the stream (a 429 is still a wait); once tokens flow, they
    are handed on as they arrive."""
    client = _load()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)

    def call(left):
        api = client if left is None else client.with_options(timeout=left)
        return api.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=messages, max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0, stream=True)

    stream = runtime.within_budget(call, _retryable(),
                                   what="the Azure endpoint")
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
