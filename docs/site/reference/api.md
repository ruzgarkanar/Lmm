# The API surface

Five methods cover ingesting and asking. Everything else this project
claims — contradiction, causality, the trust ladder, the memory dynamics —
lives on `m.session`, which is public and is the same object `Memory` is
built on. A feature with no documented entry point is not a feature.

## Memory

The full signature, with every default exactly as the code holds it:

```python
Memory(path=None,             # the store on disk; None is transient
       who="#operator",       # whose trust level a statement arrives at
       mode="STRICT",         # the gate's strictness
       cache=True,            # memoise repeated questions in-process
       persona="",            # the VOICE
       warmth=None,           # temperature of the voice surfaces
       reply_tokens=None,     # length cap of the voice surfaces
       style="",              # the FORMAT, the composer's alone
       identity=None,         # what the memory may say about itself
       encoder=None,          # the meaning channel's vectors
       dense=True,            # the meaning channel itself
       reranker="bundled")    # the ordering inside a proposal
```

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

!!! warning "Benchmarking: `cache=False` is not 'nothing is reused'"
    Below `Memory`, the engine pools its own **deterministic** calls
    (temperature 0, byte-exact prompt) for the life of the **process**,
    across `Memory` instances, and the extractor memoises sentences it has
    already read. Asking one question twice in a single process can
    therefore cost **zero calls** even with the cache off and a freshly
    built memory. This is right in production — a repeated deterministic
    call has one answer — and wrong to measure through. Give each variant
    its own process.

`warmth` and `reply_tokens` are **`None` by default**, not numbers: unset, each
voice surface keeps its own measured default, and a number here overrides all
of them at once.

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
| `m.learn(what, source=, deep=)` | teach it a file or a string. **`source="#Vendor"`** is the stamp everything from this reading carries — it is what makes a multi-document store answerable per document, and what `scope=` later selects on. `deep` mines each sentence for triples with the engine: `None` (the default) means **False for a file, True for text handed in directly**, so a document costs nothing and a typed fact costs one call per sentence. The returned `Learned` reports `.calls` | no engine for files |
| `m.ask(q, explain=True, fluent=, shape=)` | answer. `explain=True` returns an `Answer` carrying `.sources`, `.abstained`, `.engine_error`, `.route`; without it you get a plain string. `fluent=True` skips the router for a turn that is known to be conversation. **`shape=`** declares the turn's kind (`"none"`, `"count"`, `"sum"`, `"order"`, `"when"`, `"material"`) for a caller that already knows it: the engine is then asked neither what shape the message is nor whether it is about the memory itself — two calls of six, on every turn of a batch. A wrong declaration costs the organ it would have reached. **`standalone=True`** answers the question alone: the turn neither inherits the conversation's subject nor leaves one behind — what a matrix of independent cells needs, and what building a second `Memory` per cell was standing in for. **`quoted=True`** answers from ONE line the store holds and lets the store check it: ~30% fewer calls and tokens, at the cost of answers that span several lines (measured, two runs of 15 questions). **Cannot write memory** | engine |
| `m.compose(brief, seats=24, topics=, on_line=)` | a structured draft from the evidence — per-topic gathering, gated line by line, streamed to `on_line` as lines survive, returned with its sources. `seats` is how many evidence lines each topic may draw on | engine |
| `m.reset()` | end the CONVERSATION — the recent turns, the subject the last turn was about, the brief, and the documents the topic had come to be about. Nothing learned is forgotten: the graph, the evidence and the aids are untouched. A batch of independent questions wants this between cells, or `ask(..., standalone=True)` | no engine |
| `m.where(term)` | which documents mention this — names and counts, the census | no engine |
| `m.themes(least=3, most=12)` | which documents belong together, and on what entities — community detection over the store's own graph, deterministic. `least` is the smallest group worth reporting, `most` the largest number of groups | no engine |
| `m.distil(text, source=, speaker=)` | write the events AND ongoing facts a passage reports into the graph — the engine lists each as thing/what-happened/KIND, the passage's own words admit the thing, each survivor is one gated dated record; the KIND is an index key beside the bridges, never a claim (0.7) | one call per passage |
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

Two operator-set facts about the conversation itself (both optional):

| attribute | what it says |
|---|---|
| `m.session.asker = "user"` | WHO is asking — that speaker's lines seat first in the event organs; the tally that outranks an enumeration must live in this speaker's line |
| `m.session.asked_at = date(2023, 4, 18)` | WHEN the question is asked — the plan's `now` primitive reads it, so "how many days ago…" is arithmetic against this day; unset, it is the calendar's today (0.7) |

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
| `.covered` | of the question's demands (its content words), the share this answer carries — a count, no model call, nothing it can fabricate |
| `.missing` | the demands the answer did not carry: what to ask about next |
| `.engine_error` | **why** it abstained, when the reason was not the memory: `True` only when the engine could not be reached at all. An abstention with this `False` is the store's own honest "I do not hold that"; with it `True`, nothing was asked of the store at all — retry or alert, do not record a capability as absent |
| `.sources` | the provenance stamps the answer rests on |
| `.subject` | the subject label the turn was about |
| `.from_graph` | answered by the graph alone (zero model calls) |
| `.route` | which organs the turn passed through, in order — `("record",)`, `("chain", "refuse", "count")`, `("plan",)`, and `("engine-error",)` when the engine fell |

A question that asks for several things gets a partial reading for free —
`abstained` says whether anything was found, `covered` says how much, and
`missing` says what to ask about next. A caller with grades of its own
builds them from these; the library ships the measurement, not a
vocabulary:

```python
a = m.ask("Does it consider country of departure, destination, VAT ID "
          "and transport responsibility?", explain=True)
a.covered      # 0.78
a.missing      # ('transport', 'responsibility')
```

### Which mechanism keeps the promise

The promise never changes: an answer the memory does not support cannot
leave. What a caller may choose is **which mechanism enforces it**, and
what that costs.

| | how | measured cost | what it gives up |
|---|---|---|---|
| default | the engine phrases freely; a read-back and a relation check audit the claims afterwards | 4 calls a turn | — |
| `ask(..., quoted=True)` | one call answers AND names the evidence line; the STORE checks that the line is one it offered and that the answer stays inside it — word comparisons, no model | **~30% fewer calls and tokens**, −16% wall clock | answers that span several lines: a question whose answer is spread over four lines gets the one line it quoted |

In quoted mode fabrication is not judged unlikely, it is structurally
impossible: what is spoken is assembled out of a line the store holds.
When the reading cannot be trusted — no line, a line nobody offered, a
sentence stepping outside it — the turn falls back to the ordinary path,
so the worst case is one extra call and today's behaviour.

```python
a = m.ask("which interfaces does the validator use?", explain=True)
if a.engine_error:
    ...        # infrastructure: retry, alert, fail over
elif a.abstained:
    ...        # the memory genuinely does not hold it
else:
    print(a, a.sources)
```
