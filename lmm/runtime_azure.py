"""Azure OpenAI backend — for EXPERIMENTS (same-engine comparison: LMM+4o-mini vs
RAG+4o-mini → isolates the architecture's contribution). NOT the product default:
the product is the local engine (condition-1, cheap hardware); this backend is
benchmark/development only.

Contract is the SAME as runtime.generate: generate(messages, max_tokens,
temperature, system) → plain text. Selection: LMM_BACKEND=azure. Keys from .env
(outside git).
"""
import os
import threading

_CLIENT = None
_LOCK = threading.Lock()    # prevents double client init during parallel ingestion


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env():
    path = os.path.join(_root(), ".env")
    if os.path.exists(path):
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
                    api_version=os.environ["AZURE_OPENAI_API_VERSION"])
    return _CLIENT


def generate(messages, max_tokens=256, temperature=0.7, system=None):
    client = _load()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)
    out = client.chat.completions.create(
        model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        messages=messages, max_tokens=max_tokens,
        temperature=temperature if temperature > 0 else 0)
    return (out.choices[0].message.content or "").strip()
