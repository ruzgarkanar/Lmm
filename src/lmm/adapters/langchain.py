"""LMM inside a LangChain application, without LangChain inside LMM.

Two doors, because a chain can want two different things from a memory,
and only one of them keeps the promise:

`retriever(memory)` is the ordinary one — passages for a prompt, each
stamped with the document it came from. It is retrieval, so the gate is
not in play, and an application that pours these into its own LLM gets
exactly the guarantees that LLM gives: none. The stamps are still there
so the application can show its work.

`tool(memory)` is the one worth having. An agent calls it and gets back
an ANSWER THAT WAS AUDITED — with its sources, and with the abstention
intact when memory holds nothing. This is the whole difference between
LMM and a vector store, and it survives the crossing only if the "I do
not have that" comes through as itself rather than as an empty string
the agent will paper over.

LangChain is NOT a dependency. If `langchain_core` is importable the
objects are real LangChain ones; if it is not, the same calls return
duck-typed equivalents with the same attributes, so a test suite (and a
user who only wants the shape) needs nothing installed.

    from lmm import Memory
    from lmm.adapters.langchain import retriever, tool

    m = Memory("mind.lmm"); m.learn("handbook.pdf")
    chain_retriever = retriever(m)          # .invoke("...") -> [Document]
    agent_tool = tool(m)                    # .invoke({"question": "..."})
"""


class Passage:
    """What a Document is, for a caller who has no LangChain."""

    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = dict(metadata or {})

    def __repr__(self):
        return "Passage(%r, %r)" % (self.page_content[:40], self.metadata)


def _document_class():
    try:
        from langchain_core.documents import Document
        return Document
    except Exception:                                    # noqa: BLE001
        return Passage


def passages(memory, query, most=8):
    """The evidence LMM would read for this question, as documents.

    One passage per evidence line, stamped with the source the store
    attests — not with a chunk id, because a chunk id is not provenance
    and the point of this project is that a claim can be traced back to
    a document a person can open.
    """
    document = _document_class()
    session = memory.session
    lines = session.evidence.find(query, most=most)
    stamps = list(session.evidence.last_sources)
    stamps += [""] * (len(lines) - len(stamps))
    return [document(page_content=line,
                     metadata={"source": stamp, "query": query})
            for line, stamp in zip(lines, stamps)]


def retriever(memory, most=8):
    """A LangChain retriever over this memory (duck-typed if absent)."""
    try:
        from langchain_core.retrievers import BaseRetriever

        class LMMRetriever(BaseRetriever):
            memory_: object
            most_: int = 8

            def _get_relevant_documents(self, query, **_kw):
                return passages(self.memory_, query, self.most_)

        return LMMRetriever(memory_=memory, most_=most)
    except Exception:                                    # noqa: BLE001
        class _Retriever:
            def __init__(self, memory, most):
                self.memory, self.most = memory, most

            def invoke(self, query, **_kw):
                return passages(self.memory, query, self.most)

            get_relevant_documents = invoke

        return _Retriever(memory, most)


DESCRIPTION = (
    "Answer a question from the organisation's own documents. Returns the "
    "answer with the documents it rests on, and says plainly when the "
    "answer is not in memory. Prefer this over guessing.")


def answer(memory, question):
    """The audited answer, in the shape an agent can act on."""
    got = memory.ask(question, explain=True)
    return {"answer": str(got),
            "abstained": bool(getattr(got, "abstained", False)),
            "sources": list(getattr(got, "sources", ()) or ())}


def tool(memory, name="lmm_answer", description=DESCRIPTION):
    """An agent tool that returns an audited answer, never a guess."""
    def _call(question):
        got = answer(memory, question)
        # THE ABSTENTION CROSSES AS ITSELF. An agent handed an empty
        # string writes its own answer over the silence, which is the
        # exact failure this memory exists to refuse; so the sentence
        # stays, and the stamps ride with it when there are any.
        if got["abstained"] or not got["sources"]:
            return got["answer"]
        return "%s\n\nSources: %s" % (got["answer"], ", ".join(got["sources"]))

    try:
        from langchain_core.tools import StructuredTool
        return StructuredTool.from_function(
            func=lambda question: _call(question),
            name=name, description=description)
    except Exception:                                    # noqa: BLE001
        _call.name = name
        _call.description = description
        _call.invoke = lambda given, **_kw: _call(
            given["question"] if isinstance(given, dict) else given)
        return _call
