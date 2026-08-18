"""COST METER — a counter wrapped around the model clients, nothing else.

This module does not change any library behaviour. It monkey-patches, at the
SDK boundary, the two things that actually cost money or time:

  * `openai.resources.chat.completions.Completions.create` — every chat call,
    whoever makes it (LMM's `runtime_azure`, or LangChain's `AzureChatOpenAI`
    on the RAG side, which uses the same SDK underneath). The `usage` field the
    server returns is the ground truth for tokens; we do not estimate.
  * `sentence_transformers.SentenceTransformer.encode` — the RAG embedding
    step. This one spends NO API tokens at all: it is a local model burning
    CPU. That nuance is recorded explicitly (calls, texts, characters,
    seconds) rather than hidden behind a zero.

CALL ATTRIBUTION. A chat call is charged to the LMM function that made it, by
walking the stack for the nearest frame inside `src/lmm/`. That is why the
report can say WHERE the tokens burn (extract / answer / read-back /
relation-check) without a single line added to the library.
"""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LMM_SRC = os.path.join(ROOT, "src", "lmm")

# Which LMM function a call belongs to. The four buckets the report asks for:
#   extract        — reading a document/message into triples (the write path)
#   answer         — producing the sentence the user sees
#   read-back      — the verification gate re-reading what was just said
#   relation-check — the small yes/no classifiers the graph consults
_BUCKET = {
    "extract": "extract",
    "reextract": "extract",           # overridden to read-back when the gate calls it
    "answer": "answer",
    "chat": "answer",
    "identity_answer": "answer",
    "confirm": "answer",
    "refusal": "answer",
    "offer_research": "answer",
    "confirm_cause": "answer",
    "supported": "read-back",
    "answers_asked": "read-back",
    "are_rivals": "relation-check",
    "is_causal": "relation-check",
    "is_causal_question": "relation-check",
    "is_identity_question": "relation-check",
    "is_affirmative": "relation-check",
    "category_from": "relation-check",
}


class Meter:
    """One accumulator. `phase` labels what is being paid for right now."""

    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "?"
        self.rows = []          # one dict per chat call
        self.embed = []         # one dict per embedding call

    # -- recording -------------------------------------------------------
    def chat(self, **row):
        with self.lock:
            row["phase"] = self.phase
            self.rows.append(row)

    def embedding(self, **row):
        with self.lock:
            row["phase"] = self.phase
            self.embed.append(row)

    # -- reading ---------------------------------------------------------
    def slice(self, phase=None):
        return [r for r in self.rows if phase is None or r["phase"] == phase]

    def totals(self, phase=None):
        rows = self.slice(phase)
        return {
            "calls": len(rows),
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "api_seconds": round(sum(r["seconds"] for r in rows), 3),
        }

    def by_bucket(self, phase=None):
        out = {}
        for r in self.slice(phase):
            b = out.setdefault(r["bucket"], {"calls": 0, "prompt_tokens": 0,
                                             "completion_tokens": 0})
            b["calls"] += 1
            b["prompt_tokens"] += r["prompt_tokens"]
            b["completion_tokens"] += r["completion_tokens"]
        return out

    def embed_totals(self, phase=None):
        rows = [r for r in self.embed if phase is None or r["phase"] == phase]
        return {
            "calls": len(rows),
            "api_tokens": 0,            # local model — this zero is the point
            "texts": sum(r["texts"] for r in rows),
            "characters": sum(r["characters"] for r in rows),
            "cpu_seconds": round(sum(r["seconds"] for r in rows), 3),
        }


METER = Meter()


def _caller_bucket():
    """The LMM function that made this call, as a (function, bucket) pair.

    Walks outward from the SDK frame to the nearest frame living in src/lmm.
    `reextract` is ambiguous by design — the same helper serves the write path
    and the verification gate — so it is disambiguated by whether the extract
    path is anywhere below it on the stack."""
    frame, names = sys._getframe(1), []
    hit = None
    while frame is not None:
        f = frame.f_code.co_filename
        if f.startswith(_LMM_SRC):
            name = frame.f_code.co_name
            names.append(name)
            if hit is None and name in _BUCKET:
                hit = name
        frame = frame.f_back
    if hit is None:
        return ("rag", "answer")        # no LMM frame → the RAG baseline's call
    bucket = _BUCKET[hit]
    if hit == "reextract" and "extract" not in names:
        bucket = "read-back"            # the gate re-reading a spoken sentence
    return (hit, bucket)


def install():
    """Patch the SDK boundaries. Idempotent."""
    _patch_chat()
    _patch_embeddings()


def _patch_chat():
    try:
        from openai.resources.chat import completions as C
    except Exception:                                          # noqa: BLE001
        return
    target = C.Completions
    if getattr(target.create, "_metered", False):
        return
    original = target.create

    def create(self, *a, **kw):
        who, bucket = _caller_bucket()
        t0 = time.time()
        out = original(self, *a, **kw)
        dt = time.time() - t0
        usage = getattr(out, "usage", None)
        METER.chat(
            fn=who, bucket=bucket, seconds=dt,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0)
        return out

    create._metered = True
    target.create = create


def _patch_embeddings():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:                                          # noqa: BLE001
        return
    if getattr(SentenceTransformer.encode, "_metered", False):
        return
    original = SentenceTransformer.encode

    def encode(self, sentences, *a, **kw):
        items = [sentences] if isinstance(sentences, str) else list(sentences)
        t0 = time.time()
        out = original(self, sentences, *a, **kw)
        METER.embedding(texts=len(items),
                        characters=sum(len(str(s)) for s in items),
                        seconds=time.time() - t0)
        return out

    encode._metered = True
    SentenceTransformer.encode = encode
