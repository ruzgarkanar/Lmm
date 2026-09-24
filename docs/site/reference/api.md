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
       reranker="bundled",    # the ordering inside a proposal
       judge=None)            # who answers the gates' yes/no questions
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
- **`judge`** — who answers the gates' yes/no questions. Unset, this
  memory's own engine, which is what every measurement here was made
  with. See [A judgment is a seam](#a-judgment-is-a-seam).

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

### The session numbers

How hard a turn tries is a session attribute rather than a constructor
argument, because it is about effort rather than voice. The first two
default to **1**, and both were measured down from higher numbers: the
extra readings they bought stopped winning once retrieval improved. The
rest are wagers a corpus can settle for itself.

!!! note "Where a turn's calls actually go"
    Measured per organ on fifteen questions, counting only the round
    trips the deterministic pool did not absorb: **an answered turn costs
    3.5 calls and an abstaining one costs 9.0**. A fifth of the turns
    spend two fifths of the budget, and they are the ones that end up
    saying "I don't know". A turn the quoted reading carries costs
    exactly **two**: the reading and its relation gate.

| attribute | default | what raising it buys |
|---|---|---|
| `m.session.CANDIDATES` | 1 | more evidence subsets answered from, and the gate picks among them. Was 3; across eleven field questions the narrow readings won nothing, and collapsing it went from 139 calls to 104 with one more answer correct. |
| `m.session.VIEWS` | 1 | a second, wider view for each judge when the first declines. Was 2; the second view confirmed nothing on either judge in the same run. |

Raise them on a corpus where retrieval is weak — they are insurance, and
insurance is worth buying where the risk is real.

| `m.session.SEATS` | 0 (= `evidence.WINDOW`, six) | how many evidence lines the answering block seats — the largest single consumer of prompt in a turn. Measured engine-free on NIST SP 800-63B: the line carrying the gold answer arrives at **seat 1 for nine of eleven questions and seat 2 for the tenth**, and seats three to six change the miss count not at all while being two thirds of the block. With the engine, six seats against three over the same questions: **zero verdicts changed** at 42 → 40 calls and −18% prompt. The default stands because one corpus of literal questions cannot show the two risks — an answer split across two lines, and the gates reading this block as their wide view |
| `m.session.evidence.RECENT` | True | whether **when** a line was written gets a vote in retrieval. A third ranking — the same lines, newest first — handed to the same RRF fusion the other channels use, so it is an order and not a weight. Built only where **every** retrieved line carries a day, so a document store never has one: NIST's 12,253 sentences produced zero verdict changes. Measured on the conversation benchmark: paraphrase 3/4 → **4/4**. A speaker who repeats themselves will have the terse latest repetition preferred over a richer earlier telling — a limit stated in advance and not yet measured |
| `m.session.WIDEN` | **False** | whether a turn that abstained buys one more retrieval, with words the engine proposes and the store approves. Measured over three corpora and thirteen refusing turns, it rescued **one**, and that one was an off-subject answer — including on a question set with a labelled *paraphrase* class, which is what it exists for. Switching the meaning channel off did not bring it back. Set it `True` on a corpus whose questions and documents use different words for one thing |
| `m.session.QUOTED_RELATION` | True | whether a quoted turn still answers for its **relation**. Quoting replaces the read-back — the store checks the answer against lines it offered, for nothing — and replaces nothing for the other gate, because a quotation is not made relevant by being a quotation. Setting it `False` restores 0.8.1: measured on fifteen questions, the quoted reading then carries 9 turns instead of 7 at 58 calls instead of 86, and one genuinely wrong answer comes back instead of an honest abstention |

## A judgment is a seam

Both gates ask one shape of question — **does the evidence say this**
(`says`), **does this answer what was asked** (`answers`) — and both pay a
frontier engine to write "yes" in prose that is then parsed. A model class
now exists that answers exactly this shape as a typed decision with a
calibrated probability, for a fraction of the latency and the price.

This library adopts none of them. What it offers is the seam one could sit
behind:

```python
def judge(kind, question, answer, evidence):
    """kind is "says" or "answers". Return a probability, or None."""
    return my_classifier(kind, question, answer, evidence)

m = Memory("mind.lmm", judge=judge)
```

Pass nothing and the engine is asked exactly as before. Three properties
make the seam safe to hand to a stranger:

- a judge that returns **`None` has declined**, and the engine is asked —
  a provider that cannot answer must not be able to refuse an answer
  silently;
- a judge that **raises** is the same thing: an outage is not a verdict;
- **the boundary is this memory's**, not the provider's
  (`m.session.JUDGE_BOUNDARY`, the majority), so a supplier cannot move a
  gate by changing what it calls confident.

!!! warning "Where such a judge belongs, and where it does not"
    Not "wherever there is a call". A typed-decision model's declared
    failure mode is **confident wrongness inside its schema** — and on a
    gate that is a fabrication which passed the audit, which is worse
    than a prose engine's hallucination, because the hallucination is
    what the gate is shaped to catch.

    Put it where a miss costs **recall** rather than the promise, and
    above all where this memory has **no judge at all** today: ordering a
    proposal, filtering candidates before a cascade is paid for. There it
    can only add. Replacing the read-back is the last move, not the
    first, and it wants shadow numbers on your own corpus before it is
    trusted.

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

### What reading a document does to its tables

No call to make and nothing to configure — it is what `learn()` does — but
it decides what later questions can reach, so it is worth knowing.

A **row is a unit**: the cells the document put together and no others. A
table whose cells have shattered onto separate lines (the usual PDF dump)
is reassembled by the period in its cell lengths; a table written with
**delimiters** — `|` or tabs, which is markdown, ASCII tables and most of
what an ingested file contains — is already whole on one line, and where a
rule (`| --- | --- |`) follows the first row, that row names the columns
and each cell is stored as `Name: value`. No window ever spans two rows.

An **index column is dropped**. Rows ending in a bare number that never
decreases, rises across at least three rows and carries no column name
above it are a table of contents or a register: the number says where a
thing is written, not what it is. Keeping it makes a question like "what
is this regulation's number?" answerable with the page it starts on —
past every gate, because the page really is written beside the name. A
**named** column is a field and is never touched, however its values are
sorted.

If you need the page numbers themselves as data, give them a column name,
or keep that table in a separate file the memory reads as prose.

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
| `.covered` | of the question's demands (its content words), the share this answer carries — a count, no model call, nothing it can fabricate. **An abstention reports 0.0** and misses everything (0.8.1): a refusal that restates the question used to carry every demand in its own words and report 1.0 |
| `.missing` | the demands the answer did not carry: what to ask about next |
| `.unseen` | of those demands, the words **this store has never held** — in any sentence of any document. `missing` is about the answer; this is about the store, and it separates *"the document never mentions a fee"* from *"we did not find it"*. Engine-free. A paraphrase looks the same from here (asked who *heads* NIST, the document says "acting director"), so it is reported, never acted on |
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

!!! warning "`covered` is a count, not a quality score"
    It counts how many of the question's words the answer says back, so
    **an answer that restates the question scores higher than one that
    quotes the document.** Measured on one real question: a fluent
    restatement 0.88, a correct verbatim answer out of the document 0.25.
    Two consequences a caller should know. Cells answered by different
    routes are not comparable — the quoted reading's answers are terse
    and score low, the ordinary path's are phrased and score high — and
    `covered` was measured **not** to work as a relevance signal: on 44
    field cells the one wrong answer sat at 0.25, above fifteen correct
    ones, four of which sat at 0.00. No threshold separates them.

    Read it for what it is: which of the things you asked about are
    named in the reply, and which to ask about next.

    To compare two cells anyway, compare within a route: `.route` rides
    the same `Answer` object as `.covered`, so a caller that has one
    always has the other — there is no reading in which the scale is
    available and its route is not.

### Which mechanism keeps the promise

The promise never changes: an answer the memory does not support cannot
leave. What a caller may choose is **which mechanism enforces it**, and
what that costs.

| | how | measured cost | what it gives up |
|---|---|---|---|
| default | the engine phrases freely; a read-back and a relation check audit the claims afterwards | 4 calls a turn | — |
| `ask(..., quoted=True)` | ONE call answers AND names the evidence line — the router is not paid either, since the reading it routes for is already chosen; the STORE checks that the line is one it offered and that the answer stays inside it — word comparisons, no model. When a check fails, the ordinary path runs | **~30% fewer calls and tokens**, −16% wall clock | answers that span several lines: a question whose answer is spread over four lines gets the one line it quoted |
| `ask(..., quoted="only")` | the same, except the STORE's refusal stands instead of falling back — an engine that declined outright still falls back, having offered nothing to refuse | **40% fewer calls again** (54 → 32 over fifteen questions) | answers the ordinary path would have rescued: measured, it asserted nothing false where falling back spoke one wrong number, and missed three answers where falling back found one |

When the quoted reading carries a turn, that turn costs **one engine
call** — a RAG's call count, with a stamp on the answer and the store
checking it. Across fifteen questions it carried eight, and the seven
that fell back paid for the attempt as well, so the AVERAGE was 4.6
calls against the ordinary path's 6.3: −27% calls, −28% tokens, −18%
wall clock, and one question answered that the ordinary path had
abstained on. Parity per carried turn; not parity on the average.

Those figures are a matched pair measured at **0.7.12** and are the
conservative end. An integrator measuring 44 cells on two real documents
reported 9.2 calls a turn falling to 4.3 — roughly twice the saving —
with strict accuracy rising from 84.1% to 90.9% rather than falling. Our
own slice sits between them. The pair above is not restated with newer
numbers because only the quoted arm has been re-measured since, and half
a comparison is not one.

In quoted mode fabrication is not judged unlikely, it is structurally
impossible: what is spoken is assembled out of a line the store holds.
When the reading cannot be trusted — no line, a line nobody offered, a
sentence stepping outside it, or one that joins two passages — the turn
falls back to the ordinary path, so the worst case is one extra call and
today's behaviour.

!!! warning "What quoted mode trades, and for whom"
    Fabrication is structurally out; **relevance is not**, and the quoted
    reading reaches the relevance failure more easily than the ordinary
    path does, because the ordinary path judges a claim with a read-back
    and a relation check and this one does not. Reported from the field
    on 44 cells: four false-empty answers became one false-covered. That
    was strongly profitable for the reporter and would not be for a
    caller whose cost of a wrong "yes" is high.

    The commitment is also **engine-bound**. On the reported
    reproduction, `gpt-4o-mini` abstains 4 times out of 4 and `gpt-4.1`
    commits 4 out of 4: the more capable engine is the more willing to
    build an answer out of whatever is on register. Shadow the setting on
    your own corpus and engine before trusting the table above.

    Since 0.8.1 one shape of that error is closed without a model call: a
    quotation may rest on several lines, but they must be views of ONE
    passage. A window is neighbouring sentences, so where two topics meet
    it holds the end of one and the start of the next — and an answer
    that takes a clause from each never leaves its quote, which is why no
    containment check could see it. That cost **2 of 11 carried turns**
    on our own slice; the two are still answered correctly by the
    ordinary path.

```python
a = m.ask("which interfaces does the validator use?", explain=True)
if a.engine_error:
    ...        # infrastructure: retry, alert, fail over
elif a.abstained:
    ...        # the memory genuinely does not hold it
else:
    print(a, a.sources)
```
