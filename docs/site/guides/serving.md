# Serving it, and embedding it

A memory that lives in one process is a library. This page is about the
three doors out of that process — HTTP, LangChain, TypeScript — and the
one property all three are built to protect: **the answer keeps its
audit**. A caller anywhere must be able to tell a fact from a refusal,
and see the documents a claim rests on. A door that returns a bare string
turns this project into a chat completion with extra steps.

## HTTP, one memory per user

```bash
python -m lmm.serve --root ./stores --port 8000
```

```bash
curl -s localhost:8000/learn -d '{"user":"ada","text":"Ada leads R&D.","source":"#doc:team.txt"}'
curl -s localhost:8000/ask   -d '{"user":"ada","question":"who leads R&D?"}'
```

```json
{"answer": "Ada leads R&D.", "abstained": false,
 "sources": ["#doc:team.txt"], "subject": "R&D"}
```

| route | body | returns |
|---|---|---|
| `POST /ask` | `user`, `question`, `fluent?` | `answer`, `abstained`, `sources`, `subject` |
| `POST /learn` | `user`, `text` or `path`, `source?`, `deep?` | `facts`, `evidence`, `source`, `adapter`, `warnings` |
| `POST /compose` | `user`, `brief` | `text`, `sources` |
| `POST /where` | `user`, `term` | `where`: `[{source, hits}]` |
| `GET /health` | — | `ok`, `users` |

The user may also travel in an `X-LMM-User` header.

Three things this door is careful about:

- **Every user's memory is their own.** The store is chosen by the
  caller's id and nothing else; no route reads across users; an id that
  could climb a path (`../secrets`) is refused before it becomes a
  filename.
- **One writer at a time.** A session mutates a graph and an index while
  it learns, so each user's memory carries its own lock. Different users
  still answer in parallel.
- **The answer is the same answer.** This is a transport, not a second
  brain: no gate, no rephrasing, no fallback text is added.

It binds `127.0.0.1` unless you say otherwise, because a memory full of a
customer's documents should not reach a public interface because a flag
defaulted that way. Set `LMM_TOKEN` to require
`Authorization: Bearer <token>` on every request.

## TypeScript

No dependencies; `fetch` and the routes above.

```ts
import { LMM } from "@lmm/client";

const m = new LMM({ base: "http://localhost:8000", user: "ada" });
await m.learn({ text: "Ada leads R&D.", source: "#doc:team.txt" });

const a = await m.ask("who leads R&D?");
if (a.abstained) console.log("memory does not hold this");
else console.log(a.answer, a.sources);
```

`ask` resolves to an `Answer` — `answer`, `abstained`, `sources`,
`subject` — because an application that receives only prose has to
re-derive "did it actually know that?" from the words.

## LangChain

```python
from lmm import Memory
from lmm.adapters.langchain import retriever, tool

m = Memory("mind.lmm"); m.learn("handbook.pdf")

chain_retriever = retriever(m)   # .invoke("...") -> [Document], each stamped
agent_tool = tool(m)             # .invoke({"question": "..."}) -> audited answer
```

The **retriever** is ordinary retrieval: passages for someone else's
prompt, stamped with the document each came from. The gate is not in play
here, so whatever the caller's model then says about them carries the
caller's own guarantees — the stamps are there so the application can
show its work.

The **tool** is the one worth having. An agent calls it and gets back an
answer that was audited, with its sources, and gets a refusal *as
itself*: hand an agent an empty string where memory holds nothing and the
agent writes its own answer over the silence, which is the exact failure
this project exists to refuse.

LangChain is not a dependency. With `langchain_core` installed these are
real LangChain objects; without it the same calls return duck-typed
equivalents with the same attributes.
