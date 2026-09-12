# The API surface

Five methods cover ingesting and asking. Everything else this project
claims — contradiction, causality, the trust ladder, the memory dynamics —
lives on `m.session`, which is public and is the same object `Memory` is
built on. A feature with no documented entry point is not a feature.

## Memory

```python
m = Memory("mind.lmm",
           persona="You are Ada, a warm onboarding coach.",   # the VOICE
           style="One section per course: name, duration, outcomes.",  # the FORMAT
           warmth=0.6,          # temperature of the voice surfaces
           reply_tokens=1200,   # length cap of the voice surfaces
           encoder=None,        # the meaning channel's vectors
           reranker="bundled",  # the ordering inside a proposal
           dense=True)          # the meaning channel itself
```

The operator's knobs, each travelling exactly where it belongs — and none of
them reaching a verifier, because a judge whose thermostat the caller can
turn is not a judge:

- **`persona`** — the voice. Rides in front of the phrasing prompts (chat,
  answer, refusal) and stops at the document's edge: the composer never
  sees it.
- **`style`** — the document's shape, the persona's mirror twin. Travels
  ONLY to the composer's request; headings and ordering are the operator's,
  every fact beneath them is still the material's.
- **`warmth` / `reply_tokens`** — temperature and length of the same voice
  surfaces. Unset, every surface keeps its measured default.
- **`encoder`** — the meaning channel's vectors. Unset, the 30 MB static
  multilingual matrix that ships in the wheel. Any callable
  `list[str] -> list[list[float]]` replaces it and then nothing of ours is
  loaded.
- **`reranker`** — the ordering inside a proposal. `"bundled"` is late
  interaction over that same matrix; any callable
  `(query, texts) -> list[int]` replaces it; `None` turns it off.
- **`dense=False`** — no meaning channel at all. The store answers exactly as
  it did before the channel existed, and numpy is not needed.

```python
# your own encoder, and a cross-encoder doing the reordering
from sentence_transformers import SentenceTransformer, CrossEncoder
bi, cross = SentenceTransformer("BAAI/bge-m3"), CrossEncoder("BAAI/bge-reranker-v2-m3")

def order(query, texts):
    scored = cross.predict([(query, t) for t in texts])
    return sorted(range(len(texts)), key=lambda i: -scored[i])

m = Memory("mind.lmm",
           encoder=lambda texts: bi.encode(texts, normalize_embeddings=True).tolist(),
           reranker=order)
```

### Two session numbers

How hard a turn tries is a session attribute rather than a constructor
argument, because it is about effort rather than voice. Both default to **1**,
and both were measured down from higher numbers: the extra readings they
bought stopped winning once retrieval improved.

| attribute | default | what raising it buys |
|---|---|---|
| `m.session.CANDIDATES` | 1 | more evidence subsets answered from, and the gate picks among them. Was 3; across eleven field questions the narrow readings won nothing, and collapsing it went from 139 calls to 104 with one more answer correct. |
| `m.session.VIEWS` | 1 | a second, wider view for each judge when the first declines. Was 2; the second view confirmed nothing on either judge in the same run. |

Raise them on a corpus where retrieval is weak — they are insurance, and
insurance is worth buying where the risk is real.

| call | what it does | engine? |
|---|---|---|
| `m.learn(what)` | teach it a file or a string | no engine for tables |
| `m.ask(q, explain=True)` | answer, with `.sources` and `.abstained` — **cannot write memory** | engine |
| `m.compose(brief, topics=, on_line=)` | a structured draft from the evidence — per-topic gathering, gated line by line, streamed to `on_line` as lines survive, returned with its sources | engine |
| `m.where(term)` | which documents mention this — names and counts, the census | no engine |
| `m.distil(text, source=, speaker=)` | write the EVENTS a passage reports into the graph — the engine lists them, the passage's own words admit them, each survivor is one gated dated record | one call per passage |
| `m.bridge()` | teach the store, once, what words readers ask its fields with — afterwards those questions are answered by the record itself, in milliseconds | one call per field, once |
| `m.about(label)` | the records held on a concept | no engine |
| `m.facts` | how many records exist | no engine |
| `m.save(path)` | graph and evidence, both | no engine |

## Session

| call | what it does | engine? |
|---|---|---|
| `m.session.respond(msg, teach=True, on_line=)` | a conversational turn at operator trust — the surface that may teach | engine |
| `m.session.respond(msg, teach=False)` | the CONSULTATION surface — statements are context, questions search the whole conversation, a delivery request routes to the composer; cannot write memory | engine |
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

## The consultation surface

`respond(msg, teach=False)` is the customer-facing conversation — it can
never write into the operator's memory, and four behaviours exist only
there:

- a **statement** ("we are a bank, team of ten") is context: it routes to
  the chat voice, and its terms accumulate in the consultation's brief;
- a **question** searches with the whole consultation riding along — "the
  best three-hour material" still knows the topic named two turns earlier;
- a **delivery request** ("you decide, put a programme together") goes
  straight to the composer and returns a source-stamped catalogue built
  from the brief;
- **echoing the user's own words** back is conversation, not assertion —
  the gate lets a consultant repeat what the customer just said, while a
  word the customer never said still answers to the graph.

Pass `on_line=` to `respond` or `compose` and the draft **streams**: each
line is judged the moment its newline arrives and handed to the callback
while the engine still writes. A refused line is never seen; a backend
that cannot stream degrades to batch by itself.

The *no engine* column is not a footnote. Those calls are plain Python over
a dict — microseconds, no network, no key — which is why a spreadsheet loads
in milliseconds and why the reasoning half of this system costs nothing to
run.


## Over HTTP, and in other runtimes

`python -m lmm.serve --root ./stores` exposes `/ask`, `/learn`,
`/compose`, `/where` and `/health`, one memory per user, with the
abstention and the source stamps intact on the wire.
`lmm.adapters.langchain` offers a stamped retriever and an agent tool
that returns audited answers; `sdk/typescript` is a dependency-free
client whose `ask` resolves to the same four fields as the Answer object
below. See [Serving and embedding](../guides/serving.md).

## Answer object

`ask(..., explain=True)` returns a string that additionally carries what the
turn knows about itself:

| attribute | meaning |
|---|---|
| `.abstained` | did this turn assert anything — the structural stamp, in any language |
| `.sources` | the provenance stamps the answer rests on |
| `.subject` | the subject label the turn was about |
| `.from_graph` | answered by the graph alone (zero model calls) |
| `.route` | which organs the turn passed through, in order — `("record",)`, `("chain", "refuse", "count")`, `("plan",)` |
