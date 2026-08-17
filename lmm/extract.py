"""Extraction: message → {kind, triples}. Qwen extracts but does NOT write — the
extracted triples enter the GATE as candidates (session/verify decides). Replaces
reader.py; the contract stays the same (kind + triples) so session logic doesn't break.
"""
import json
import unicodedata

from v3.dataset import fold
from lmm import prompts, runtime


def _fold(text):
    """fold + invisible-combining-mark cleanup. Qwen CAN produce sequences like
    'i̇çecek' (i + U+0307) in its JSON — the invisible dot opens a separate graph
    node, chains silently break. First NFC (real accents merge into one code
    point: ç/ö/ü/é preserved), then the REMAINING combining marks are dropped
    (those that couldn't merge are this kind of garbage). Not a language rule —
    Unicode's own tables."""
    text = unicodedata.normalize("NFC", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return fold(text)

WRITE, ASK, CHAT = "WRITE", "ASK", "CHAT"


def _json(raw):
    """Pulls the FIRST COMPLETE JSON object from Qwen's output — balanced-brace scan.

    The old greedy `\\{.*\\}` swallowed everything between the first { and the
    last }: if Qwen gives an explanation plus a second {...} after the fact
    (common in small models), the two objects merged into invalid JSON and
    extraction was silently zeroed out. Now it stops where the first opening
    brace balances.
    """
    start = raw.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except ValueError:
                    return {}
    return {}


def _clean(triples):
    """Normalizes triples to (subject, predicate, value); drops subject-less ones.

    Qwen may give a triple both as a list [[s,p,v]] and as a dict
    [{"subject":..}] (dicts are likely with chat models) — both are accepted,
    otherwise a taught fact would silently drop.

    The accepted KEY NAMES are the ones the prompt asks for, and nothing else.
    Turkish aliases used to sit beside them ("özne"/"yüklem"/"değer"), because
    a Turkish prompt sometimes came back with Turkish keys; with the prompt in
    the codebase's own language that cannot happen, and keeping them would mean
    one user language had a private channel into the parser that German and
    Spanish did not.
    """
    out = []
    for t in triples or []:
        if isinstance(t, dict):
            subject = t.get("subject") or ""
            predicate = t.get("predicate") or t.get("relation") or ""
            value = t.get("value") or t.get("object") or ""
            parts = [str(subject).strip(), str(predicate).strip(),
                     str(value).strip()]
        elif isinstance(t, (list, tuple)):
            parts = [str(x).strip() for x in (list(t) + ["", "", ""])[:3]]
        else:
            continue
        subject, predicate, value = parts
        if subject:
            # _fold, NOT .lower(): 'İçecek'.lower() → 'i̇çecek' (invisible
            # U+0307) was opening a separate node and breaking chains; _fold
            # cleans both this and the combining marks Qwen produces ready-made.
            out.append((_fold(subject), _fold(predicate), _fold(value)))
    return out


def _malformed(subject, value):
    """Has the subject SWALLOWED the sentence — if the subject of "a lion is a
    mammal" becomes 'a lion is a mammal', the node label becomes a sentence and
    "what is a lion" can never find it (silent loss). Structural criterion: the
    value's words are INSIDE the subject. Not a language rule — a set view."""
    if not (subject and value):
        return False
    sw = set(subject.split())
    vw = set(value.split())
    return bool(vw) and vw <= sw and len(sw) > len(vw)


def extract(message):
    """Read the message → {'kind': WRITE|ASK|CHAT, 'triples': [(s,p,v)]}.
    Deterministic (temperature 0): same sentence, same extraction.

    SAFETY: a subject-swallowed-the-sentence triple is first repaired with
    reextract (a different prompt, more robust); if that too yields nothing, the
    triple DROPS — honestly failing to learn rather than opening a garbage node
    and saying "learned" (the write-side face of fabrication-0)."""
    raw = runtime.generate(message, system=prompts.EXTRACT_SYSTEM,
                           max_tokens=160, temperature=0.0)
    data = _json(raw)
    kind = str(data.get("kind", CHAT)).upper()
    if kind not in (WRITE, ASK, CHAT):
        kind = CHAT
    triples = _clean(data.get("triples"))
    if kind == WRITE and any(_malformed(s, v) for s, _p, v in triples):
        repaired = reextract(message)
        if repaired:
            triples = repaired
        else:
            triples = [t for t in triples if not _malformed(t[0], t[2])]
    return {"kind": kind, "triples": triples}


def reextract(sentence):
    """For the verification gate: extracts the FACT claims a sentence carries.
    If there is no fact, [] (greeting, opinion, 'I don't know')."""
    raw = runtime.generate(sentence, system=prompts.REEXTRACT_SYSTEM,
                           max_tokens=120, temperature=0.0)
    return _clean(_json(raw).get("triples"))
