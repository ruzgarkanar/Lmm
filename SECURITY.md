# Security Policy

## Supported versions

LMM is at `0.1.0` and pre-1.0. Fixes land on the latest release only.

## Reporting a vulnerability

Please report privately, not in a public issue:

- GitHub → **Security → Report a vulnerability** (preferred), or
- email the maintainer at the address on the GitHub profile.

Include what you did, what happened, and what you expected. A proof of concept
helps. Expect an acknowledgement within a week; please give a fix a reasonable
window before disclosing.

## What is in scope

- Code execution, path traversal or resource exhaustion triggered by a
  **document LMM is asked to ingest**. `learn()` parses untrusted files through
  third-party readers, and that is the largest attack surface in the project.
- Reading or writing files outside the memory path a caller supplied.
- A **fabrication that passes the gate** — an answer asserting a fact the
  memory does not support. This is a correctness bug in the ordinary case, but
  the gate is a safety property of this system, so treat a general bypass as a
  security issue and report it privately.
- Leaking a memory's contents across sessions or callers that should not share
  it.

## What is not in scope

- The **language model's** own output quality, refusals or biases. LMM wraps a
  third-party engine; problems in the engine belong to the engine.
- Answers that are wrong but honestly sourced, and abstentions on questions the
  memory cannot support. Those are documented limits, not vulnerabilities — see
  the README's "Honest limits".
- Anything requiring the attacker to already control the machine, the memory
  file or the model weights.

## Handling your own data

LMM writes what it learns to disk in readable form: the graph at your memory
path and the evidence index in `<path>.evidence` beside it. **Sentences from
ingested documents are stored verbatim**, because provenance is the point — an
answer has to be able to show the line it rests on.

So the memory file is as sensitive as the documents that went into it. Do not
commit it, and note that `Memory()` without a path keeps everything in process
memory only. Conversation logs, where enabled, contain verbatim user input;
this repository's `.gitignore` excludes `logs/`, `*.lmm` and `*.evidence` for
that reason.

Engine credentials are read from the environment. Keep them in `.env`, which is
gitignored, and never in code.
