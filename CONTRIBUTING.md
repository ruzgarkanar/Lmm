# Contributing to LMM

Thank you for looking. This file is short on process and long on one thing: the
two rules that make this project different from a retrieval pipeline. A change
that breaks either of them will be declined however much it improves a score,
and knowing that up front will save you an afternoon.

## Running things

```bash
git clone https://github.com/ruzgarkanar/lmm
cd lmm
pip install -e .                   # the core: no dependencies at all
python3.11 tests/test_core.py      # 63 tests, no model, no GPU, no network
```

The test suite is plain python — no pytest, no fixtures, no config. It runs in
seconds and needs nothing installed, because the whole discrete layer is
counters, links and thresholds. If your change cannot be tested that way, that
is worth a conversation before it is worth a patch.

Optional readers, when you are working on ingestion:

```bash
pip install -e '.[pdf,xlsx,docx]'
```

## Running the benchmark

```bash
pip install -e '.[local]'                        # or [gguf], or [azure]
python3.11 benchmarks/lmm_side.py \
    benchmarks/corpus.txt benchmarks/questions.json /tmp/result.json
python3.11 benchmarks/score.py benchmarks/questions.json /tmp/result.json
```

`corpus.txt`, `corpus_en.txt` and `corpus_es.txt` are the same invented world in
three languages — the same facts in the same order, the same 17 questions, the
same invented names. That is deliberate: no answer in any language can come from
what a model already knows, and a change that helps one language and hurts
another shows up immediately.

**These benchmarks are noisy. A single run is not a measurement.** Report the
median of at least three samples of the same configuration, and say so. Several
"improvements" in this project's history were one lucky seed.

**Gold-in-top-K is necessary but not sufficient.** It has passed unchanged while
end-to-end scores dropped two points, because it asks "is the answer in the top
six" and not "did the retrieved block change". Measure both.

The real-document benchmarks quoted in the README are not in this repository.
They ran against a third party's product manual, an institution's internal
strategy document and an inspection report naming real people. Those are not
ours to redistribute; only the aggregate scores are published.

## The two rules

### 1. No document-specific constants

Every threshold must be **derived from the material at hand** or stated as what
it means. A number chosen while looking at one PDF is not a parameter, it is a
memory of that PDF, and it will be paid for on every other document.

This is not hypothetical. An audit of this codebase found a 700-character record
window, a 40-character table cell, a 0.75 similarity and an inflection rule with
a special case for short spec-table subjects — all of them fitted, none of them
justifiable. Removing them, along with few-shot examples that quoted the test
document's own voltages, cost **62/63 → 59/63**. That gap was the share of the
score that came from having seen the test, and giving it up was the point.

So, concretely:

- ❌ `if len(cell) <= 40:` — where did 40 come from?
- ✅ `if len(cell) <= _cell_bound(cells):` — the split in *this sheet's* own
  cell lengths.
- ❌ a few-shot example quoting a real document's part number.
- ✅ a few-shot example from an invented world.

If you genuinely need a constant, say in the code what it means and what would
have to be true for it to be wrong.

### 2. No hand-written language rules

No word lists, no stopword sets, no phrase tables, no "if the question starts
with…", no per-language special cases — anywhere in the pipeline. Users converse
in their own language and the system's own sentences are engine-generated in
that language.

This rule was not free either. Three things had quietly made Turkish the home
language: every answer was framed by hand-written Turkish labels, every few-shot
example in every classifier was Turkish, and the training-data builder pasted
subjects into a Turkish question form. The labels are structural field names
now, the examples span four languages, and the question form travels with the
dataset whose language it belongs to. Removing the last of it cost one more
point, on a question the engine now honestly declines.

The **measuring tools** are bound by this too, and that is the subtler half. To
score whether the system abstained, the benchmark used to match Turkish refusal
phrases — which would have scored every honest English refusal as a fabrication
and reported a language dependency the engine does not have. The judgement moved
inside instead: a turn records whether it ASSERTED anything, decided by
re-extracting the claims from what was spoken. Tests K1–K6 defend this. If your
change needs to recognise a phrase in a particular language, it is the wrong
change.

Format cues are fine — a colon is a separator, not a language rule. Unicode's own
tables are fine. A list of words you wrote is not.

## Sending a change

- Open an issue first for anything structural. For a bug, the measurement that
  shows it is worth more than the fix.
- Small diffs. One idea per pull request.
- Every test must pass: `python3.11 tests/test_core.py`.
- **Add a test.** Every test in `tests/test_core.py` is a pathology that was
  actually measured on this code, written back as an assertion so it cannot
  return. Follow that: name the defect, not the function.
- Explain in the commit message what you measured, including what it cost. This
  project's history records rejected experiments as carefully as accepted ones,
  because "we already tried that, here are the numbers" is the most useful thing
  a codebase can tell the next person.
- If you change behaviour, say so plainly. Silent behaviour changes are the one
  thing review cannot catch.

## Licence

Contributions are accepted under the Apache License 2.0 (see `LICENSE`). By
opening a pull request you agree your work is licensed under those terms.
