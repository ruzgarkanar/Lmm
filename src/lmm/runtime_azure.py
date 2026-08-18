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
        out = api.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=messages, max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0)
        return (out.choices[0].message.content or "").strip()

    return runtime.within_budget(call, _retryable(), what="the Azure endpoint")
