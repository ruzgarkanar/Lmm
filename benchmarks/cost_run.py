"""COST RUN — one measured sample of one configuration.

    python3.11 benchmarks/cost_run.py <side> <corpus> <questions> <out.json>

    side: rag | lmm-shallow | lmm-deep

Both sides run on the SAME engine (Azure gpt-4o-mini) so the numbers compare
ARCHITECTURES, not model sizes — the same discipline README's accuracy table
uses. Every token comes from the server's own `usage` field via meter.py; none
is estimated.

Writes a JSON with: ingestion cost, per-question cost (with the LMM call-type
breakdown), and how many questions were answered with ZERO model calls.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import meter                                                    # noqa: E402


def _env():
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


def _question_rows(questions, run_one):
    """Ask each question under its own meter phase, so per-question cost and
    the zero-call count are facts rather than an average."""
    rows = []
    for i, q in enumerate(questions):
        phase = f"q{i}"
        meter.METER.phase = phase
        t0 = time.time()
        said = run_one(q["soru"])
        wall = time.time() - t0
        t = meter.METER.totals(phase)
        rows.append({"soru": q["soru"], "cevap": said,
                     "wall_seconds": round(wall, 3),
                     "buckets": meter.METER.by_bucket(phase), **t})
    return rows


def run_rag(corpus_path, questions):
    _env()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_openai import AzureChatOpenAI
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    meter.install()          # after the imports exist, before anything runs

    class STEmbed(Embeddings):
        def __init__(self):
            self.m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        def embed_documents(self, texts):
            return self.m.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return self.m.encode([text], normalize_embeddings=True)[0].tolist()

    corpus = open(corpus_path, encoding="utf-8").read()
    meter.METER.phase = "ingest"
    t0 = time.time()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80).split_text(corpus)
    store = Chroma.from_texts(chunks, STEmbed())
    ingest_wall = time.time() - t0

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        temperature=0)

    def one(question):
        docs = store.similarity_search(question, k=4)
        context = "\n".join(d.page_content for d in docs)
        return llm.invoke([
            ("system", "Answer using ONLY the information in the given "
                       "context. If it is not in the context, say that you do "
                       "not know; never guess. Answer briefly, and ONLY in "
                       "the SAME LANGUAGE as the question."),
            ("user", f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
        ]).content

    rows = _question_rows(questions, one)
    return {"ingest": {"wall_seconds": round(ingest_wall, 3),
                       "chunks": len(chunks),
                       **meter.METER.totals("ingest"),
                       "embedding": meter.METER.embed_totals("ingest")},
            "query_embedding": meter.METER.embed_totals(),
            "questions": rows}


def run_lmm(corpus_path, questions, deep):
    _env()
    os.environ["LMM_BACKEND"] = "azure"          # same engine as the RAG side
    os.chdir(ROOT)
    from lmm.session import Session
    from lmm.mind import Mind

    meter.install()

    corpus = open(corpus_path, encoding="utf-8").read()
    s = Session(None)                            # transient memory, no disk
    m = Mind(s)
    meter.METER.phase = "ingest"
    t0 = time.time()
    wrote, skipped = s.learn_text(corpus, source="#doc:corpus", deep=deep)
    m.run()                                      # symbolic derivation, no model
    ingest_wall = time.time() - t0
    derived = sum(1 for r in s.memory.records.values() if r.source == "#inference")

    rows = _question_rows(questions, s.respond)
    return {"ingest": {"wall_seconds": round(ingest_wall, 3),
                       "facts": wrote, "skipped": skipped, "derived": derived,
                       **meter.METER.totals("ingest"),
                       "buckets": meter.METER.by_bucket("ingest"),
                       "embedding": {"calls": 0, "api_tokens": 0,
                                     "cpu_seconds": 0.0}},
            "questions": rows}


def main():
    side, corpus_path, q_path, out_path = sys.argv[1:5]
    questions = json.load(open(q_path, encoding="utf-8"))
    if side == "rag":
        data = run_rag(corpus_path, questions)
    elif side in ("lmm-shallow", "lmm-deep"):
        data = run_lmm(corpus_path, questions, deep=(side == "lmm-deep"))
    else:
        raise SystemExit(f"unknown side: {side}")

    qs = data["questions"]
    data["side"] = side
    data["corpus"] = os.path.basename(corpus_path)
    data["zero_call_questions"] = sum(1 for r in qs if r["calls"] == 0)
    data["question_count"] = len(qs)
    data["query_totals"] = {
        "calls": sum(r["calls"] for r in qs),
        "prompt_tokens": sum(r["prompt_tokens"] for r in qs),
        "completion_tokens": sum(r["completion_tokens"] for r in qs),
        "wall_seconds": round(sum(r["wall_seconds"] for r in qs), 3),
    }
    json.dump(data, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"{side} {os.path.basename(corpus_path)}: "
          f"ingest {data['ingest']['calls']} calls / "
          f"{data['ingest']['prompt_tokens']}+{data['ingest']['completion_tokens']} tok / "
          f"{data['ingest']['wall_seconds']}s | "
          f"query {data['query_totals']['calls']} calls over {len(qs)} q, "
          f"zero-call {data['zero_call_questions']} → {out_path}", flush=True)


if __name__ == "__main__":
    main()
