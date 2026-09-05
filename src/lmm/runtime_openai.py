"""OpenAI-compatible backend — the one most people already have a key for.

Selection: `LMM_BACKEND=openai`. Contract is the SAME as `runtime.generate`:
generate(messages, max_tokens, temperature, system) -> plain text.

WHY THIS EXISTS BESIDE `runtime_azure`. The Azure client needs a deployment
name, a resource endpoint and an api-version, which is a shape only Azure has;
everyone else — OpenAI itself, OpenRouter, Groq, Together, a vLLM or Ollama
server on the next machine — speaks the plain chat-completions API. Those are
one client with a different `base_url`, so they are one file rather than one
file each, and adding a provider is a URL rather than a patch.

    OPENAI_API_KEY=sk-...                      # OpenAI
    LMM_MODEL=gpt-4o-mini

    OPENAI_BASE_URL=https://openrouter.ai/api/v1   # anything else
    OPENAI_API_KEY=...
    LMM_MODEL=meta-llama/llama-3.3-70b-instruct

    OPENAI_BASE_URL=http://localhost:8000/v1       # your own server
    OPENAI_API_KEY=unused

NOTHING ABOUT THE ARCHITECTURE CHANGES WITH THE ENGINE. The graph, the gate and
the derivation are plain python and never call this file; what an engine
decides is how a retrieved record is worded, not whether it may be spoken. A
weaker engine here loses fluency, not provenance.

Keys are read from the environment or from a `.env` outside git — this file
never holds one.
"""
import os
import threading

from lmm import paths, runtime

_CLIENT = None
_LOCK = threading.Lock()    # prevents double client init during parallel ingestion


def _env():
    path = paths.find_env()
    if path:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k, v)


def model():
    """Which model to ask. `LMM_MODEL` names it; the default is the small
    hosted model this project's cost tables are measured against, so a reader
    comparing their own bill against `benchmarks/COST.md` is comparing the
    same thing unless they chose otherwise."""
    return os.environ.get("LMM_MODEL") or os.environ.get(
        "OPENAI_MODEL", "gpt-4o-mini")


def _load():
    global _CLIENT
    if _CLIENT is None:
        with _LOCK:
            if _CLIENT is None:     # double-checked: calls are thread-safe (httpx)
                _env()
                from openai import OpenAI              # noqa: PLC0415
                key = os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise RuntimeError(
                        "LMM_BACKEND=openai needs OPENAI_API_KEY. Set it in the "
                        "environment or in a .env file (which .gitignore already "
                        "excludes). For a server that wants no key — a local "
                        "vLLM or Ollama — set it to any non-empty string and "
                        "point OPENAI_BASE_URL at the server.")
                _CLIENT = OpenAI(
                    api_key=key,
                    # None keeps the SDK's own default (api.openai.com), so a
                    # plain OpenAI key needs this variable not to exist rather
                    # than to be set to something correct.
                    base_url=os.environ.get("OPENAI_BASE_URL") or None,
                    # THE RETRIES LIVE IN `runtime.within_budget` — the same
                    # reason as the Azure client: two retry policies with two
                    # clocks multiply, and one measured question once took 613
                    # seconds that way. Retry while there is budget, not one
                    # round longer.
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
            model=model(), messages=messages, max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0)
        return (out.choices[0].message.content or "").strip()

    return runtime.within_budget(call, _retryable(), what="the OpenAI endpoint")


def generate_stream(messages, max_tokens=256, temperature=0.7, system=None):
    """The streamed twin of `generate` — same client, same budget discipline
    on OPENING the stream; once tokens flow, they are handed on as they
    arrive."""
    client = _load()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)

    def call(left):
        api = client if left is None else client.with_options(timeout=left)
        return api.chat.completions.create(
            model=model(), messages=messages, max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0, stream=True)

    stream = runtime.within_budget(call, _retryable(),
                                   what="the OpenAI endpoint")
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
