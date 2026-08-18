#!/bin/sh
# Full cost matrix: 3 sides x 2 languages x 3 repeats, run STRICTLY SEQUENTIALLY.
# Sequential matters: both sides share one rate-limited Azure deployment, so
# running them concurrently would inflate each other's wall clock and make the
# seconds column meaningless. Token counts would survive that; seconds would not.
cd "$(dirname "$0")/.." || exit 1
mkdir -p benchmarks/cost
for rep in 1 2 3; do
  for lang in tr en; do
    if [ "$lang" = tr ]; then
      C=benchmarks/corpus.txt;    Q=benchmarks/questions.json
    else
      C=benchmarks/corpus_en.txt; Q=benchmarks/questions_en.json
    fi
    for side in rag lmm-shallow lmm-deep; do
      OUT="benchmarks/cost/${side}_${lang}_${rep}.json"
      [ -f "$OUT" ] && { echo "skip $OUT"; continue; }
      python3.11 benchmarks/cost_run.py "$side" "$C" "$Q" "$OUT" 2>&1 | tail -2
    done
  done
done
echo "MATRIX DONE"
