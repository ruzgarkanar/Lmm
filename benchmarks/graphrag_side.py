"""GraphRAG side — Microsoft's own package, not a reimplementation.

WHY THIS EXISTS. The RAG baseline in `rag_side.py` is the industry-standard
embedding pipeline, and beating it says something — but it is not the system
this project's claim actually competes with. Graph-structured memory is an
active field: Microsoft GraphRAG, Zep/Graphiti, Mem0, Letta. Publishing "we
beat chunk-and-embed" while the nearest neighbour goes unmeasured is the kind
of gap a reader finds in the first comment.

So this side runs `pip install graphrag` — the real package, its own indexer,
its own prompts, its own default config. Nothing here reimplements GraphRAG's
method, which means the result cannot be dismissed as a strawman.

THE SAME DISCIPLINE AS EVERY OTHER SIDE:

    same engine        Azure gpt-4o-mini, exactly what rag_side and the LMM
                       side use, so the column compares architectures
    same questions     the file the other sides are given
    same corpus        the file the other sides are given
    same output shape  {ingest_ms, cevaplar:[{soru, cevap, ms}]} — which is
                       what `evaluate.py` scores, unchanged and side-blind

LOCAL SEARCH, NOT GLOBAL. GraphRAG offers two query modes and they answer
different questions. Global search summarises communities and is built for
"what are the themes across this corpus"; local search is entity-focused and
is the right mode for the factual lookups these benchmarks ask ("how many
characters shall a memorized secret be"). Scoring GraphRAG's global mode on
factual questions would understate it, and this comparison is only worth
running if the other side is set up to win.

COST WARNING — read before running. GraphRAG's indexer is expensive: it
extracts entities and relationships from every text unit, then summarises
communities. On a 1.6 KB corpus that is a handful of calls; on a full standard
like NIST SP800-63B it is hundreds. The indexer caches, so a re-run over an
unchanged corpus is nearly free, but the first pass is not.

Runs in its own virtualenv (`benchmarks/.graphrag-venv`) because graphrag pulls
in roughly a hundred packages — spacy, lancedb, litellm — and none of them
belong anywhere near a library whose core install has no dependencies at all.

Usage:
    benchmarks/.graphrag-venv/bin/python benchmarks/graphrag_side.py \\
        benchmarks/corpus_en.txt benchmarks/questions_en.json \\
        benchmarks/result_graphrag.json
"""
import asyncio
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

# GraphRAG reads its credentials from its own names. The repository keeps one
# .env with the Azure names every other side uses; this maps between them so
# there is one place to change a key, not two.
SETTINGS = """
completion_models:
  default_completion_model:
    model_provider: azure
    model: {deployment}
    azure_deployment_name: {deployment}
    api_base: {endpoint}
    api_version: {version}
    api_key: ${{GRAPHRAG_API_KEY}}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: azure
    model: {embed_deployment}
    azure_deployment_name: {embed_deployment}
    api_base: {endpoint}
    api_version: {version}
    api_key: ${{GRAPHRAG_API_KEY}}
    retry:
      type: exponential_backoff

input_storage:
  type: file
  base_dir: "input"

output_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

cache:
  type: json
  storage:
    type: file
    base_dir: "cache"

vector_store:
  type: lancedb
  db_uri: output/lancedb
  # The store's default width is 3072 (text-embedding-3-large). The embedding
  # deployment available here is text-embedding-3-small at 1536, so the table
  # was created 3072 wide while every query vector arrived 1536 wide, and each
  # search failed with a dimension mismatch AFTER the index had been paid for.
  # The width has to be declared next to the model that produces it.
  vector_size: {vector_size}

embed_text:
  embedding_model_id: default_embedding_model

extract_graph:
  completion_model_id: default_completion_model
  entity_types: [organization,person,geo,event]
  max_gleanings: 1

summarize_descriptions:
  completion_model_id: default_completion_model
  max_length: 500

community_reports:
  completion_model_id: default_completion_model

local_search:
  chat_model_id: default_completion_model
  embedding_model_id: default_embedding_model
"""


def _env():
    """Load the repository's .env the way every other side does."""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key, value)


def _prepare(corpus_path, root):
    """A GraphRAG project directory: settings, input, credentials.

    Rebuilt from scratch when the corpus CHANGES, so a previous document's
    index can never leak into this one's answers — the same reason every other
    side starts from an empty memory. When the corpus is byte-identical the
    directory is kept, because the indexer's cache lives inside it and
    re-indexing an unchanged document is money spent for nothing.
    """
    inside = os.path.join(root, "input", "corpus.txt")
    wanted = open(corpus_path, "rb").read()
    same = (os.path.exists(inside) and open(inside, "rb").read() == wanted)
    if os.path.exists(root) and not same:
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "input"), exist_ok=True)
    if not same:
        shutil.copy(corpus_path, inside)

    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    embed = os.environ.get("AZURE_OPENAI_EMBED_DEPLOYMENT",
                           "text-embedding-3-small")
    settings = SETTINGS.format(
        deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        embed_deployment=embed,
        vector_size=1536 if "small" in embed else 3072,
        endpoint=endpoint,
        version=os.environ["AZURE_OPENAI_API_VERSION"])
    with open(os.path.join(root, "settings.yaml"), "w", encoding="utf-8") as h:
        h.write(settings)
    with open(os.path.join(root, ".env"), "w", encoding="utf-8") as h:
        h.write(f"GRAPHRAG_API_KEY={os.environ['AZURE_OPENAI_API_KEY']}\n")
    return same


def _tables(root):
    """The parquet files the indexer leaves behind, as local_search wants."""
    import pandas as pd

    out = os.path.join(root, "output")
    wanted = ("entities", "communities", "community_reports", "text_units",
              "relationships")
    held = {}
    for name in wanted:
        path = os.path.join(out, f"{name}.parquet")
        held[name] = pd.read_parquet(path) if os.path.exists(path) else None
        if held[name] is None:
            print(f"  missing table: {name}.parquet", flush=True)
    return held


def main():
    _env()
    # ABSOLUTE PATHS, resolved before graphrag is imported. The indexer changes
    # the process working directory, so a relative path handed in on the
    # command line stops resolving halfway through the run — the questions file
    # went missing after a 50-second index had already been paid for.
    corpus_path = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 \
        else os.path.join(HERE, "corpus.txt")
    questions_path = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 \
        else os.path.join(HERE, "questions.json")
    out_path = os.path.abspath(sys.argv[3]) if len(sys.argv) > 3 \
        else os.path.join(HERE, "result_graphrag.json")
    questions = json.load(open(questions_path, encoding="utf-8"))

    from graphrag import api
    from graphrag.config.load_config import load_config

    # TOKEN METERING. GraphRAG records its INDEXING usage in stats.json, but
    # the query path's calls are not written anywhere with their token counts,
    # and a cost comparison missing the per-question column is not a cost
    # comparison. graphrag 3.x speaks to the model through litellm, which
    # offers a success callback carrying the provider's own usage object —
    # so the figure below is the server's count, the same discipline
    # `benchmarks/meter.py` uses on the other sides.
    # TOKEN METERING AT THE HTTP LAYER — the only seam that sees everything.
    #
    # Two higher seams were tried and both under-counted, silently, which is
    # the dangerous kind of measurement error: litellm's success callback and
    # graphrag's own metrics processor each reported 17 calls and 93 prompt
    # tokens — the query EMBEDDINGS only, no completions at all. The chat path
    # is async and routes around both.
    #
    # Every one of them is still an HTTPS request whose response carries the
    # provider's own `usage` object, so counting there cannot miss a call by
    # taking a different code path. This is the server's count, which is the
    # same discipline `benchmarks/meter.py` applies to the other sides.
    import json as _json

    import httpx

    usage = {"calls": 0, "prompt": 0, "completion": 0}

    # GRAPHRAG STREAMS ITS ANSWERS, and a streaming response only carries a
    # usage object when the caller asks for one (`stream_options.include_usage`,
    # which graphrag does not set). So for chat calls the server's count does
    # not exist to be read — the first three metering attempts all returned the
    # embeddings alone for exactly this reason.
    #
    # The REQUEST is not streamed, so the prompt side is counted exactly from
    # the payload with the model's own tokenizer. Completions are counted from
    # the text that came back. This is labelled an estimate wherever it is
    # reported, and it is the prompt column that carries the cost anyway.
    import tiktoken

    _encoding = tiktoken.get_encoding("o200k_base")

    def _count(response):
        try:
            payload = _json.loads(response.content)
        except Exception:                                   # noqa: BLE001
            payload = None
        if isinstance(payload, dict):
            held = payload.get("usage")
            if isinstance(held, dict):
                prompt = held.get("prompt_tokens") or 0
                completion = held.get("completion_tokens") or 0
                if prompt or completion:
                    usage["calls"] += 1
                    usage["prompt"] += prompt
                    usage["completion"] += completion
                    return
        # Streamed chat call: count the request we sent.
        request = getattr(response, "request", None)
        if request is None or b"chat/completions" not in (request.url.raw_path
                                                          or b""):
            return
        try:
            sent = _json.loads(request.content)
        except Exception:                                   # noqa: BLE001
            return
        text = "".join(str(one.get("content") or "")
                       for one in sent.get("messages", ()))
        usage["calls"] += 1
        usage["prompt"] += len(_encoding.encode(text))
        usage["estimated"] = usage.get("estimated", 0) + 1

    _send = httpx.Client.send
    _asend = httpx.AsyncClient.send

    def _patched(self, *args, **kwargs):
        response = _send(self, *args, **kwargs)
        _count(response)
        return response

    async def _apatched(self, *args, **kwargs):
        response = await _asend(self, *args, **kwargs)
        _count(response)
        return response

    httpx.Client.send = _patched
    httpx.AsyncClient.send = _apatched

    root = os.path.join(HERE, ".graphrag-work")
    reused = _prepare(corpus_path, root)
    config = load_config(root)

    started = time.time()
    asyncio.run(api.build_index(config))
    ingest_ms = round((time.time() - started) * 1000)
    print(f"index built in {ingest_ms} ms"
          f"{' (corpus unchanged, cache reused)' if reused else ''}",
          flush=True)

    tables = _tables(root)
    # Indexing usage is separated from query usage: the index is cached, so a
    # re-run over an unchanged corpus makes no indexing calls and everything
    # counted after this line belongs to the questions.
    indexing = dict(usage)
    results = []
    for question in questions:
        started = time.time()
        try:
            answer, _ = asyncio.run(api.local_search(
                config=config,
                entities=tables["entities"],
                communities=tables["communities"],
                community_reports=tables["community_reports"],
                text_units=tables["text_units"],
                relationships=tables["relationships"],
                covariates=None,
                community_level=2,
                response_type="Single sentence",
                query=question["soru"]))
        except Exception as reason:                         # noqa: BLE001
            answer = f"[error] {reason}"
        took = round((time.time() - started) * 1000)
        text = answer if isinstance(answer, str) else str(answer)
        results.append({"soru": question["soru"], "cevap": text, "ms": took})
        print(f"> {question['soru']}\n  {text[:160]}   ({took}ms)", flush=True)

    asked = {key: usage[key] - indexing.get(key, 0) for key in usage}
    print(f"tokens — indexing {indexing}   questions {asked}", flush=True)
    json.dump({"ingest_ms": ingest_ms, "cevaplar": results,
               "usage_indexing": indexing, "usage_questions": asked},
              open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
