Thank you for the patch. Three questions, and the first one decides the rest.

## What does this change, and what did it measure?

<!-- One or two sentences. If the change touches the answer path, the
     benchmark numbers before and after belong here — median of three, because
     a single run of these benchmarks is not a measurement. -->

## Which of the two rules does it touch?

`CONTRIBUTING.md` names two rules that make this a memory rather than a
retrieval pipeline:

- [ ] **The engine cannot write records.** Nothing a model produces reaches
      memory without passing `gate.admit`.
- [ ] **The engine cannot speak unsupported facts.** Nothing reaches the user
      without surviving the verification chain.
- [ ] This change touches neither.

A patch that relaxes either will be declined however much it improves a score.
If you believe a rule is wrong, open an issue first — that is a conversation
worth having, and it is a bigger one than a diff.

## Checks

- [ ] `python3.11 tests/test_core.py` passes — plain python, no pytest, no
      network, no GPU.
- [ ] New behaviour has a test in the same style. If it cannot be tested that
      way, say so here and why.
- [ ] The core install still has **no dependencies**. Anything new belongs
      behind an optional extra.
- [ ] No hand-written language anywhere in the pipeline — no phrase lists, no
      per-language rules, no templates. Behaviour comes from data or from
      counting.

## Anything you want a second opinion on?

<!-- Optional. A patch that names its own weak spot gets read more carefully,
     not less. -->
