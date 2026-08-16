"""Azure OpenAI backend — DENEY amaçlı (aynı-motor kıyası: LMM+4o-mini vs
RAG+4o-mini → mimarinin katkısını izole eder). Ürün varsayılanı DEĞİL: ürün
yerel motor (condition-1, ucuz donanım); bu backend yalnız benchmark/geliştirme.

Sözleşme runtime.generate ile AYNI: generate(messages, max_tokens, temperature,
system) → düz metin. Seçim: LMM_BACKEND=azure. Anahtarlar .env'den (git-dışı).
"""
import os
import threading

_CLIENT = None
_LOCK = threading.Lock()    # paralel yutmada çift client init'i engeller


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
            if _CLIENT is None:     # double-checked: çağrılar thread-safe (httpx)
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
