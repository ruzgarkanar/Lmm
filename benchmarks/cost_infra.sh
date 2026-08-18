#!/bin/sh
# INFRASTRUCTURE COST — measured, not asserted.
#
# Two clean virtualenvs, one per architecture, then count what actually landed
# on disk. This is the honest form of "RAG needs an embedding model and a
# vector DB; LMM needs a file": both claims become a package count and a
# megabyte figure anyone can reproduce with the same two commands.
#
# The RAG side's embedding WEIGHTS are downloaded at first use, not by pip, so
# they are measured separately from the HuggingFace cache.
set -e
REPO=$(cd "$(dirname "$0")/.." && pwd)
BASE="${1:?usage: cost_infra.sh <workdir>}"
mkdir -p "$BASE"
cd "$BASE"

measure() {           # measure <name> <venv-dir>
  SITE=$(find "$2/lib" -maxdepth 2 -name site-packages | head -1)
  echo "=== $1"
  echo "packages: $("$2/bin/pip" list --format=freeze 2>/dev/null | wc -l | tr -d ' ')"
  echo "site-packages bytes: $(du -sk "$SITE" | cut -f1)"
  echo "site-packages human: $(du -sh "$SITE" | cut -f1)"
}

if [ ! -d venv_lmm ]; then
  python3.11 -m venv venv_lmm
  ./venv_lmm/bin/pip -q install --upgrade pip >/dev/null 2>&1 || true
  # The repository itself, which is what `pip install lmm` publishes. Installed
  # from source so the measurement is of THIS tree, not of whatever is on PyPI.
  ./venv_lmm/bin/pip -q install "$REPO" >/dev/null
fi
measure "LMM core (pip install lmm)" venv_lmm

if [ ! -d venv_rag ]; then
  python3.11 -m venv venv_rag
  ./venv_rag/bin/pip -q install --upgrade pip >/dev/null 2>&1 || true
  ./venv_rag/bin/pip -q install langchain-chroma langchain-openai \
      langchain-text-splitters sentence-transformers >/dev/null
fi
measure "RAG stack (langchain-chroma + langchain-openai + langchain-text-splitters + sentence-transformers)" venv_rag

echo "=== embedding weights (downloaded at first use, not by pip)"
CACHE="$HOME/.cache/huggingface/hub"
if [ -d "$CACHE" ]; then
  du -sh "$CACHE"/*paraphrase-multilingual-MiniLM* 2>/dev/null || echo "not cached here"
fi
echo "INFRA DONE"
