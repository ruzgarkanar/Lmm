# The API surface

Five methods cover ingesting and asking. Everything else this project
claims — contradiction, causality, the trust ladder, the memory dynamics —
lives on `m.session`, which is public and is the same object `Memory` is
built on. A feature with no documented entry point is not a feature.

## Memory

| call | what it does | engine? |
|---|---|---|
| `m.learn(what)` | teach it a file or a string | no engine for tables |
| `m.ask(q, explain=True)` | answer, with `.sources` and `.abstained` — **cannot write memory** | engine |
| `m.compose(brief)` | a structured draft from the evidence — blended, gated line by line, returned with its sources | engine |
| `m.where(term)` | which documents mention this — names and counts, the census | no engine |
| `m.about(label)` | the records held on a concept | no engine |
| `m.facts` | how many records exist | no engine |
| `m.save(path)` | graph and evidence, both | no engine |

## Session

| call | what it does | engine? |
|---|---|---|
| `m.session.respond(msg, teach=True)` | a conversational turn at operator trust — the surface that may teach | engine |
| `m.session.learn_rows(rows)` | `[{column: value}]` straight to the graph | no engine |
| `m.session.learn_cause(a, b)` | record that a causes b | no engine |
| `m.session.root_causes(x)` | walk the causal chain back | no engine |
| `m.session.causes_of(x)` / `.effects_of(x)` | one step either way | no engine |
| `m.session.tension()` | the contradictions it is holding | no engine |
| `m.session.curiosity()` | what it has been asked and cannot answer | no engine |
| `m.session.sleep()` | fade, reinforce, settle episodic into semantic | no engine |
| `m.session.verdict(old, new)` | arbitrate two rival values | engine |

A record is a `core.memory.Record`: `.subject`, `.predicate`, `.value` are
concept **keys**, not strings — one spelling can be two entities, one entity
can carry labels in several languages. Read labels through
`m.session.memory.identities`. `.sources` and `.trust` are the provenance
the gate reads on the way out.

The *no engine* column is not a footnote. Those calls are plain Python over
a dict — microseconds, no network, no key — which is why a spreadsheet loads
in milliseconds and why the reasoning half of this system costs nothing to
run.

## Answer object

`ask(..., explain=True)` returns a string that additionally carries what the
turn knows about itself:

| attribute | meaning |
|---|---|
| `.abstained` | did this turn assert anything — the structural stamp, in any language |
| `.sources` | the provenance stamps the answer rests on |
| `.subject` | the subject label the turn was about |
| `.from_graph` | answered by the graph alone (zero model calls) |
