"""Scoring — objective, keyword-based (NO judge model: the judge would be
RAG's own engine, hence biased).

direct/multi-hop/causal: does one of the gold words appear in the answer (fold).
absence: does the answer ABSTAIN — something like "I don't know" AND no
        fabricated value. Criterion: we don't scan the answer for "confident
        claim" patterns absent from the corpus/gold; the practical proxy: pass
        if an abstention word is present, fail if absent and the answer is
        short-and-assertive.

Note: JSON field names ('soru', 'altin', 'tip', 'cevap', 'ms', 'ingest_ms',
'cevaplar') are Turkish because the benchmark data files are Turkish — they
are part of the data format and must not be renamed here.
"""
import json
import os
import re
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))   # benchmarks/

# Turkish abstention patterns — the FALLBACK, and no longer the criterion.
#
# Whether a sentence DECLINES TO ANSWER is a fact about meaning, and nothing in
# the answer's shape distinguishes "the manual does not state it" from "the
# manual states it". Read from OUTSIDE, that leaves only the wording, and a
# wording list is a list in ONE language: measured against an English or a
# German document, every honest abstention this system produced would have been
# scored as a fabricated answer, and the benchmark would have reported a
# language dependency the engine does not have.
#
# So the judgement moved to the only place that does not have to guess: the
# system itself. `Session.last_abstained` says whether the turn ASSERTED
# ANYTHING — raised by the single refusal door and, for the answers that
# decline without ever reaching it, by re-extracting the claims out of what was
# actually spoken. `bench/lmm_side.py` writes that stamp into the result file
# as "abstained", and `declined()` below reads it first. The patterns stay for result files that carry no stamp — the RAG
# baseline's, and every historical run already on disk — and for those the old
# structural care still applies: ordered word STEMS inside a bounded window, so
# morphology and inserted adverbs cost nothing.
#
# Each entry is a sequence of STEMS, not a phrase to find verbatim.
ABSTAIN = ("bilmiyorum", "bilgi yok", "bilgim yok", "bilgi sahip değil", "bilemem",
           "bulunmuyor", "yer almıyor", "belirtilmemiş", "bilinmiyor",
           "bilmem", "ulaşamadım", "emin değil", "yoktur")

# How many words may be INSERTED into a pattern before it stops being that
# pattern. Two is what "a hedging adverb or two" means ("bu bilgiye HENÜZ sahip
# değilim", "bu konuda ŞU ANDA bilgim yok"); it is a bound on interruption, not
# a tuned score.
SLACK = 2


def fold(t):
    t = unicodedata.normalize("NFC", t)
    return "".join(ch.casefold()[0] for ch in t if not unicodedata.combining(ch))


def _words(text):
    return re.findall(r"\w+", fold(text), re.UNICODE)


def _claim(answer):
    """The part of the answer that CLAIMS something, with the engine's own
    annotations removed.

    An answer may carry a provenance note the engine appended rather than
    asserted: a trailing parenthetical hedge and a '#source' tag ("Bilmiyorum.
    (Bu bilgi kesin değildir; #pdf:M30_70 User Manual_compressed.pdf)"). Those
    are the answer's FOOTNOTE. Measured, and this is why the distinction is
    here: reading them as claims turned 24 genuine abstentions across five
    historical result files into losses, purely because a filename has digits
    and capitals in it. Format only — parentheses and a '#' tag — no language.

    The cost is stated plainly: a value that appears ONLY inside a parenthesis
    is invisible to the value check. No answer in the record does that, and the
    abstention-pattern condition still has to hold either way."""
    text = unicodedata.normalize("NFC", answer)
    text = re.sub(r"\([^)]*\)", " ", text)
    return " ".join(w for w in text.split() if not w.startswith("#"))


def _claims_a_value(answer):
    """Does this sentence assert something SPECIFIC — a figure or a code.

    Structural, no language in it: a digit, or a token that carries a capital
    letter after its first character ('IPX7', 'DICOM', 'BF'), which is how a
    designation is written in any Latin-script document. An answer that names
    a value is answering, whatever else it hedges around it — and an ABSENCE
    question is exactly the one where naming a value is the failure."""
    for token in re.findall(r"\w+", _claim(answer), re.UNICODE):
        if any(ch.isdigit() for ch in token):
            return True
        if len(token) > 1 and any(ch.isupper() for ch in token[1:]):
            return True
    return False


def abstains(answer):
    """Does the answer DECLINE to answer.

    Two conditions, and the interesting one is the second. (1) One of the
    ABSTAIN patterns appears as an ordered run of word stems within a bounded
    window — a literal-substring test broke on the most ordinary Turkish
    sentence there is, "Bu bilgiye HENÜZ sahip değilim", which is a genuine
    abstention scored as a loss (it cost a real measurement a point); the
    window makes every such insertion free without lengthening the list.
    (2) The sentence names no specific value, because an answer that supplies a
    figure to a question the document cannot answer is a fabrication, not an
    abstention, no matter how politely it is framed."""
    if _claims_a_value(answer):
        return False
    toks = _words(answer)
    for pattern in ABSTAIN:
        stems = _words(pattern)
        limit = len(stems) + SLACK
        for start in range(len(toks)):
            i = start
            for stem in stems:
                while i < len(toks) and not toks[i].startswith(stem):
                    i += 1
                if i >= len(toks):
                    break
                i += 1
            else:
                if i - start <= limit:
                    return True
    return False


def declined(row):
    """Did this turn DECLINE to answer — the language-independent reading.

    The system's own stamp first ("abstained", written by the engine that took
    the refusal path), the wording heuristic only when there is no stamp. The
    value check survives the stamp on purpose: an "abstention" that names a
    figure is a fabrication however the engine classified its own path, and
    that half of the criterion is structural in every language.
    """
    answer = row.get("cevap") or ""
    if "abstained" in row:
        return bool(row["abstained"]) and not _claims_a_value(answer)
    return abstains(answer)


def score(path, questions):
    data = json.load(open(path, encoding="utf-8"))
    by_q = {r["soru"]: r for r in data["cevaplar"]}
    rows, per_type = [], {}
    for q in questions:
        r = by_q[q["soru"]]
        ans = fold(r["cevap"] or "")
        if q["altin"] is None:
            ok = declined(r)
        else:
            ok = any(fold(g) in ans for g in q["altin"])
        rows.append((q["tip"], q["soru"], ok, r["ms"]))
        t = per_type.setdefault(q["tip"], [0, 0, 0])
        t[0] += ok; t[1] += 1; t[2] += r["ms"]
    return data, rows, per_type


def main():
    qpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "questions.json")
    questions = json.load(open(qpath, encoding="utf-8"))
    global RAG_PATH, LMM_PATH

    for name, path in (("RAG (langchain+embedding+gpt-4o-mini)",
                        sys.argv[2] if len(sys.argv) > 2 else "bench/result_rag.json"),
                       ("LMM (graph+gate+local-3B)", sys.argv[3] if len(sys.argv) > 3 else "bench/result_lmm.json")):
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f"{name}: no results ({path})"); continue
        data, rows, per_type = score(full, questions)
        print(f"\n=== {name} ===")
        for qtype, ((ok, n, ms)) in ((k, v) for k, v in sorted(per_type.items())):
            print(f"  {qtype:10}  {ok}/{n}   avg {round(ms/n)}ms")
        top = sum(v[0] for v in per_type.values())
        n = sum(v[1] for v in per_type.values())
        print(f"  TOTAL       {top}/{n}   ingest {data.get('ingest_ms','?')}ms")
        for qtype, question, ok, ms in rows:
            if not ok:
                print(f"    ✗ [{qtype}] {question}")


if __name__ == "__main__":
    main()
