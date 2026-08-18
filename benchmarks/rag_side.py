"""RAG side — industry-standard pipeline, set up honestly and strong:
langchain text-splitter → multilingual embedding → Chroma → top-k → gpt-4o-mini
(Azure). Prompt follows best practice: "answer only from the context,
otherwise say you don't know". The engine is FAR stronger than ours (cloud
4o-mini vs local 3B) — the comparison is tilted against us; if a gap still
shows up, the claim is solid.

The system prompt used to be Turkish, because the first benchmark corpus was;
that made the BASELINE language-dependent too, and a baseline that only works
on one language cannot be run against the English and Spanish corpora the LMM
side is now measured on. It is now an English instruction that binds the reply
to the QUESTION's language — the same discipline the LMM side's ANSWER_SYSTEM
carries. The RAG numbers recorded in README were measured with the older
Turkish prompt and have NOT been re-run since; they are the Turkish-corpus
baseline and are labelled as such.

The output JSON field names ('soru', 'cevap', 'ms', 'ingest_ms', 'cevaplar')
match the benchmark data format and must not be renamed.

Output: bench/sonuc_rag.json  [{soru, cevap, ms}]
"""
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))   # benchmarks/


def _env():
    with open(os.path.join(ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v)


def main():
    _env()
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_openai import AzureChatOpenAI
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import SentenceTransformer

    class STEmbed(Embeddings):
        def __init__(self):
            self.m = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        def embed_documents(self, texts):
            return self.m.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            return self.m.encode([text], normalize_embeddings=True)[0].tolist()

    import sys
    corpus = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "corpus.txt"),
                  encoding="utf-8").read()
    t0 = time.time()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80).split_text(corpus)
    store = Chroma.from_texts(chunks, STEmbed())
    ingest_ms = round((time.time() - t0) * 1000)
    print(f"ingest: {len(chunks)} chunks, {ingest_ms}ms", flush=True)

    llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        temperature=0)

    questions = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "questions.json"),
                               encoding="utf-8"))
    results = []
    for q in questions:
        t0 = time.time()
        docs = store.similarity_search(q["soru"], k=4)
        context = "\n".join(d.page_content for d in docs)
        msg = llm.invoke([
            ("system", "Answer using ONLY the information in the given "
                       "context. If it is not in the context, say that you do "
                       "not know; never guess. Answer briefly, and ONLY in "
                       "the SAME LANGUAGE as the question."),
            ("user", f"CONTEXT:\n{context}\n\nQUESTION: {q['soru']}"),
        ])
        ms = round((time.time() - t0) * 1000)
        results.append({"soru": q["soru"], "cevap": msg.content, "ms": ms})
        print(f"> {q['soru']}\n  {msg.content}   ({ms}ms)", flush=True)

    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "result_rag.json")
    json.dump({"ingest_ms": ingest_ms, "cevaplar": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
