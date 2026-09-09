"""One harness, any document: what does LMM actually do with THIS file?

    python3.11 benchmarks/document_probe.py FILE_OR_DIR [--ask 12]

Every document this project has been tested against needed its own
throwaway script — a corpus of training outlines, a standards PDF, a
spreadsheet, a novel — and each one measured whatever that script's
author remembered to measure. This is the one that does not have to be
rewritten: it takes a file or a folder, builds the memory the way a user
would, INVENTS ITS OWN QUESTIONS FROM THE DOCUMENT, and reports what
each part of the system did with them.

WHY THE QUESTIONS ARE GENERATED. Questions written by the person who
also wrote the fixes measure the fixes. These come from the document's
own structure — a head it carries, a head the collection repeats that
this document never mentions, a value that is the largest of its field —
so the gold is mechanical and the set is as fair as the document is.

WHAT IS MEASURED, and each line says which organ it belongs to:

    ingestion     what it cost to read the file at all
    record        a field of a named document, read from the row
    census        a field question naming no document, read across all
    extremes      the largest/smallest of a numeric field
    comparison    two documents' values for one field, side by side
    absent        a field this document does not carry — must abstain
    nonsense      something no document could hold — must abstain
    rephrased     the same factual question in words the document
                  does not use (the frontier: retrieval's own limit)
    language      a refusal is spoken in the language it was asked in
    provenance    every claim carries a stamp naming its document
    cost          model calls, prompt tokens and seconds per question

The scorecard prints one line per group. What matters is not a single
number but WHICH groups fail: an abstention on `absent` is the promise
working, an abstention on `record` is retrieval failing, and a claim on
`nonsense` is the one result that would end the project.
"""
import argparse
import glob
import json
import os
import random
import re
import statistics
import sys
import time
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from lmm import evidence, runtime                           # noqa: E402
from lmm.api import Memory                                  # noqa: E402

READABLE = (".txt", ".md", ".pdf", ".docx", ".xlsx", ".csv")


def _plain(text):
    text = unicodedata.normalize("NFD", (text or "").lower()
                                 .replace("İ", "i").replace("I", "ı"))
    return "".join(c for c in text if not unicodedata.combining(c))


def _hit(answer, golds):
    plain = _plain(answer)
    return any(_plain(str(g)) in plain for g in golds)


def _gold_of(value):
    """The shortest part of a written value an answer must contain.

    A whole record line as gold scores a correct answer wrong for
    rephrasing its tail, which measures the scorer. A number with its
    unit is what the document committed to; without digits, the longest
    word carries the meaning.
    """
    pairs = re.findall(r"(\d+)\s*([^\W\d_]+)?", value, re.UNICODE)
    if pairs:
        return [" ".join(x for x in pairs[0] if x).strip()]
    words = sorted(re.findall(r"[^\W\d_]{4,}", value, re.UNICODE), key=len)
    return [words[-1]] if words else [value[:20]]


class Probe:
    """A memory, a generated question set, and a scorecard."""

    def __init__(self, target, ask=12, seed=7, identity="Assistant"):
        self.target = target
        self.ask = ask
        self.rng = random.Random(seed)
        self.identity = identity
        self.memory = None
        self.rows = {}          # group -> [(ok, question, answer, note), ...]
        self.stats = {}

    # ---------------------------------------------------------------- read
    def ingest(self):
        files = ([self.target] if os.path.isfile(self.target)
                 else sorted(f for f in glob.glob(
                     os.path.join(self.target, "*"))
                     if f.lower().endswith(READABLE)))
        if not files:
            sys.exit("no readable document in %s" % self.target)
        self.memory = Memory(None, identity=self.identity,
                             persona="Answer briefly and only from the "
                                     "documents.")
        t0 = time.time()
        warned = []
        for path in files:
            report = self.memory.learn(path)
            warned += list(report.warnings)[:1]
        store = self.memory.session.evidence
        self.stats.update(
            files=len(files), seconds=round(time.time() - t0, 1),
            units=len(store.sentences), words=len(store.index),
            sources=len(store.by_source), warnings=warned[:2])
        return self

    # ------------------------------------------------------------ question
    @staticmethod
    def _record_like(head, value):
        """Is this pair a RECORD, or a colon that happened to be in prose?

        A blank form carries labels with no values ("Your signature:"), a
        novel carries dialogue with colons ("Darcy," she cried: ...), and
        a question built on either is a question about nothing — measured,
        six of six on a tax form and six of six on a novel, with the
        memory abstaining correctly every time. A record's head is short,
        made of plain words, and its value says something.
        """
        words = re.findall(r"[^\W\d_]+", head, re.UNICODE)
        return (words and len(words) <= 3
                and len(head) <= 40
                and not re.search(r"[\[\]\"”“(),;]", head)
                and len(value) >= 2 and len(value) <= 120)

    def _records(self):
        """head -> {source: value}, over the store's own record rows."""
        store = self.memory.session.evidence
        limit, _t = store._record_bounds()
        cap = max(80, limit // 2)
        out = {}
        for text, origin in store.sentences:
            for head, value in evidence.record_pairs(text, cap):
                if self._record_like(head, value):
                    out.setdefault(head, {}).setdefault(origin, value)
        return {h: v for h, v in out.items() if v}

    def _name(self, source):
        return evidence._source_name(source)

    def _table_questions(self, path):
        """Record questions from a table FILE, asked the way a person asks.

        A spreadsheet's rows arrive in the store glued into windows, so
        reading a row's identity back out of the evidence gives subjects
        like "745 · Survived: 1 · Pclass: 3". The file itself has the
        answer: a header row names the fields, and one column holds what
        each row IS. That column is the one whose values are text and
        (nearly) unique — no list of names anywhere, just the shape of a
        table.
        """
        import csv                                          # noqa: PLC0415
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            return []
        columns = list(rows[0])
        def _identifying(col):
            values = [r.get(col) or "" for r in rows]
            texty = sum(1 for v in values if v and not v.replace(".", "").isdigit())
            return texty > len(values) * 0.9 and len(set(values)) > len(values) * 0.9
        keys = [c for c in columns if _identifying(c)]
        if not keys:
            return []
        key = keys[0]
        out = []
        for row in self.rng.sample(rows, min(len(rows), self.ask * 3)):
            subject = (row.get(key) or "").strip()
            if not subject or len(subject) > 60:
                continue
            for col in columns:
                # THE ROW'S ID IS NOT A FIELD OF IT. A table's first
                # column is usually the row's own number, which the graph
                # holds as the row's NAME rather than as one of its
                # records — "what is the PassengerId of Taylor" is asking
                # a row for its own identity, and no memory answers that
                # from a record. Measured: eight such questions, eight
                # abstentions, and nothing wrong with the memory.
                if col == key or col == columns[0]:
                    continue
                value = (row.get(col) or "").strip()
                if value:
                    out.append(("What is the %s of %s?" % (col, subject),
                                _gold_of(value)))
                    break
            if len(out) >= self.ask:
                break
        return out

    def build(self):
        """Questions from the document, one group at a time."""
        store = self.memory.session.evidence
        records = self._records()
        fields = {h: v for h, v in records.items() if len(v) >= 2}
        asked = {}

        # record: a head this document carries.
        #
        # WHO THE QUESTION IS ABOUT DEPENDS ON THE FILE'S SHAPE. In a
        # folder of documents, one document IS the subject ("what is the
        # DURATION of the Alpha Course"). In a table, one ROW is the
        # subject and the file is just where it lives — asking "what is
        # the Fare of titanic.csv" is not a question anybody would ask,
        # and measured, it is not one the memory can answer either. A
        # unit carrying three or more records is a row: its first value
        # names it, and the rest are its fields.
        pool = []
        limit, _t = store._record_bounds()
        cap = max(80, limit // 2)
        # A FIELD IS A HEAD THE DOCUMENT REPEATS — the library's own rule,
        # borrowed here so the questions are about fields rather than
        # about colons. A novel's dialogue produces "Darcy," she cried:'
        # and a form produces a label with no value; neither repeats, and
        # neither is a record. Measured: without this the novel produced
        # six questions about nothing and the memory abstained on all six,
        # which measured the generator, not the memory.
        repeats = {}
        for text, _origin in store.sentences:
            for head, _value in evidence.record_pairs(text, cap):
                repeats[head] = repeats.get(head, 0) + 1
        for text, origin in store.sentences:
            pairs = evidence.record_pairs(text, cap)
            pairs = [(h, v) for h, v in pairs
                     if self._record_like(h, v)
                     and repeats.get(h, 0) >= max(2, len(store.sentences) // 100)]
            if len(pairs) >= 3:
                subject = pairs[0][1]
                for head, value in pairs[1:]:
                    pool.append((head, subject, value))
            else:
                for head, value in pairs:
                    pool.append((head, self._name(origin), value))
        self.rng.shuffle(pool)
        seen_q = set()
        asked["record"] = []
        if os.path.isfile(self.target) and self.target.lower().endswith(".csv"):
            asked["record"] = self._table_questions(self.target)
            pool = []
        for head, subject, value in pool:
            question = "What is the %s of %s?" % (head, subject)
            if question in seen_q or len(subject) > 60:
                continue
            seen_q.add(question)
            asked["record"].append((question, _gold_of(value)))
            if len(asked["record"]) >= self.ask:
                break

        # census + extremes: a field several documents answer, asked of none
        numeric = [h for h, per in fields.items()
                   if len(per) >= 3
                   and sum(1 for v in per.values() if re.search(r"\d", v)) >= 3]
        asked["census"], asked["extremes"] = [], []
        for head in numeric[:2]:
            per = fields[head]
            asked["census"].append(("Which documents state a %s?" % head,
                                    [self._name(s) for s in list(per)[:2]]))
            best = max(per.items(),
                       key=lambda kv: max([int(d) for d in
                                           re.findall(r"\d+", kv[1])] or [0]))
            asked["extremes"].append(
                ("Which one has the highest %s?" % head,
                 [self._name(best[0])] + _gold_of(best[1])))

        # comparison: two documents, one field
        asked["comparison"] = []
        for head, per in list(fields.items())[:2]:
            if len(per) >= 2:
                (s1, v1), (s2, v2) = list(per.items())[:2]
                asked["comparison"].append(
                    ("Do %s and %s state the same %s?"
                     % (self._name(s1), self._name(s2), head),
                     _gold_of(v1) + _gold_of(v2) + ["same", "differ",
                                                    "aynı", "fark"]))

        # absent: a head the collection repeats and this document lacks,
        # with none of the head's words anywhere in it
        asked["absent"] = []
        for head, per in fields.items():
            hw = evidence._words(head)
            for src in list(store.by_source)[:40]:
                if src in per:
                    continue
                spoken = set()
                for sid in store.by_source[src]:
                    spoken |= set(evidence._words(store.sentences[sid][0]))
                if not any(w in spoken for w in hw):
                    asked["absent"].append(
                        ("What is the %s of %s?" % (head, self._name(src)),
                         None))
                    break
            if len(asked["absent"]) >= max(3, self.ask // 3):
                break

        # RECORDS OR PROSE, DECIDED BY THE DOCUMENT. A head that opens a
        # hundredth of the document's units is a field; one that opens
        # three lines of a novel is a colon. Where no head clears that,
        # the record group is empty and the prose group takes over — the
        # harness asking the kind of question the document can answer,
        # rather than the kind it knows how to generate.
        #
        # prose: a document with no records is not a document with no
        # facts. Where the record channel finds nothing (a novel, an
        # essay, a judgment), a sentence carrying a NAME is the unit of
        # meaning, and the question is asked in that sentence's own words
        # — the fair version of a paraphrase test lives in `rephrased`.
        asked["prose"] = []
        if len(asked["record"]) < 3:
            names = {}
            for text, _origin in store.sentences:
                for name in re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]{3,}"
                                       r"(?: [A-ZÇĞİÖŞÜ][a-zçğıöşü]{3,})?\b",
                                       text):
                    names.setdefault(name, []).append(text)
            rare = [(n, t) for n, t in names.items() if 1 <= len(t) <= 4]
            self.rng.shuffle(rare)
            for name, texts in rare[:self.ask]:
                sentence = texts[0]
                rest = [w for w in re.findall(r"[^\W\d_]{5,}", sentence,
                                              re.UNICODE)
                        if w not in name]
                if not rest:
                    continue
                asked["prose"].append(
                    ("What does the document say about %s?" % name,
                     [max(rest, key=len)]))
                if len(asked["prose"]) >= max(3, self.ask // 2):
                    break

        # nonsense: nothing any document could hold
        asked["nonsense"] = [
            ("What is the wifi password printed in this document?", None),
            ("Which football club is named in these documents?", None),
            ("What is the mobile phone number of the author?", None)]

        # rephrased: the same factual question, in other words
        asked["rephrased"] = []
        for question, gold in asked["record"][:3]:
            # THE FRONTIER GROUP IS BUILT BY THE ENGINE, deliberately: a
            # rephrasing written by the same hand that wrote the fix would
            # be a rephrasing the fix already covers. The question is
            # handed to the engine and asked for again in other words —
            # the document's own vocabulary is exactly what it must NOT
            # keep.
            try:
                other = runtime.generate(
                    question, max_tokens=40, temperature=0.0,
                    system=("Rewrite the user's question so that it asks "
                            "for the SAME thing in DIFFERENT words: use "
                            "synonyms, change the verb, keep any proper "
                            "name unchanged. Output only the rewritten "
                            "question."))
            except Exception:                                # noqa: BLE001
                other = ""
            if other and other.strip() != question:
                asked["rephrased"].append((other.strip(), gold))
        self.questions = asked
        return self

    # ----------------------------------------------------------------- run
    def run(self):
        times, meter = [], None
        try:
            import meter as _meter                           # noqa: PLC0415
            _meter.install()
            _meter.METER.__init__()
            meter = _meter
        except Exception:                                    # noqa: BLE001
            meter = None
        stamped = claimed = 0
        for group, items in self.questions.items():
            for question, gold in items:
                t0 = time.time()
                answer = self.memory.ask(question, explain=True)
                times.append(time.time() - t0)
                text = str(answer)
                if gold is None:
                    ok = bool(answer.abstained)
                else:
                    ok = _hit(text, gold) and not answer.abstained
                if not answer.abstained:
                    claimed += 1
                    stamped += bool(answer.sources)
                self.rows.setdefault(group, []).append(
                    (ok, question, text[:90], "" if answer.abstained
                     else ",".join(answer.sources)[:40]))
        self.stats["seconds_per_question"] = round(statistics.median(times), 2)
        self.stats["provenance"] = "%d/%d" % (stamped, claimed)
        if meter is not None:
            total = meter.METER.totals()
            n = max(1, sum(len(v) for v in self.questions.values()))
            self.stats["calls_per_question"] = round(total["calls"] / n, 1)
            self.stats["prompt_tokens_per_question"] = round(
                total["prompt_tokens"] / n)
        return self

    def language_check(self):
        """A refusal is spoken in the language it was asked in."""
        probes = [("What is the serial number of the missing part?", "en"),
                  ("Welche Seriennummer hat das fehlende Teil?", "de"),
                  ("Eksik parçanın seri numarası nedir?", "tr")]
        got = []
        for question, tag in probes:
            said = str(self.memory.ask(question))
            got.append((tag, said[:60]))
        self.stats["refusals"] = got
        return self

    # -------------------------------------------------------------- report
    def report(self):
        print("\n=== %s" % self.target)
        print("ingestion : %(files)d file(s) · %(sources)d source(s) · "
              "%(units)s units · %(words)s words · %(seconds)s s"
              % {**self.stats, "units": f"{self.stats['units']:,}",
                 "words": f"{self.stats['words']:,}"})
        for warning in self.stats.get("warnings", []):
            print("  warning : %s" % warning[:90])
        print("cost      : %s calls/q · %s prompt tokens/q · %s s/q"
              % (self.stats.get("calls_per_question", "?"),
                 self.stats.get("prompt_tokens_per_question", "?"),
                 self.stats.get("seconds_per_question", "?")))
        print("provenance: %s claims carry a source stamp"
              % self.stats.get("provenance", "?"))
        for tag, said in self.stats.get("refusals", []):
            print("refusal %s: %s" % (tag, said))
        print()
        for group, rows in self.rows.items():
            ok = sum(1 for r in rows if r[0])
            print("%-11s %2d/%-2d  %s" % (group, ok, len(rows),
                                          "" if ok == len(rows) else "<-- look"))
            for good, question, answer, stamp in rows:
                if not good:
                    print("    %-58s %s" % (question[:58], answer[:60]))
        return self


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="a document, or a folder of them")
    parser.add_argument("--ask", type=int, default=12,
                        help="how many record questions to generate")
    parser.add_argument("--identity", default="Assistant")
    args = parser.parse_args(argv)
    (Probe(args.target, ask=args.ask, identity=args.identity)
     .ingest().build().run().language_check().report())


if __name__ == "__main__":
    main()
