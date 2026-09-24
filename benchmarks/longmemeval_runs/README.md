# LongMemEval runs

Raw output of `benchmarks/longmemeval.py`, one file per run, kept so a
number in a release note can be checked rather than believed. This is the
thing that was missing: the project measured itself on this benchmark
twice before and could not do it a third time, because the harness and
the question list lived outside the repository.

**The configuration is part of the measurement.** Every file's name says
which data variant and which caller settings produced it, and the
harness prints them in its own header. A figure quoted without them
means nothing — the same thirty questions score 10% or 23% on the same
code depending on how the caller asks.

| file | variant | caller | judged |
|---|---|---|---|
| `0.12.1_s_30.json` | `longmemeval_s` (~50 sessions) | default | 3/30 |
| `0.12.1_s_30_ask_quoted.json` | `longmemeval_s` | `shape="ask", quoted=True` | 7/30 |
| `0.12.1_oracle_30.json` | `longmemeval_oracle` (evidence sessions only) | default | 13/30 |

Two readings, both in the files: `strict` is whether the gold string
appears in the answer, `judge` is an engine asked whether the answer says
what the gold says — the second is what published LongMemEval figures
use. Neither is called accuracy on its own.

The slice is deterministic: questions sorted by id, taken round-robin
across the six types, so `--most 30` is the same thirty questions on
every machine and in every release.
