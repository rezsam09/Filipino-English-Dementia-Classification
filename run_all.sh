#!/usr/bin/env bash
# run_all.sh — Execute all notebooks in order using papermill.
#
# Usage:
#   ./run_all.sh [DATA_DIR]
#
# DATA_DIR  Path to the directory containing english.csv and filipino.csv.
#           Defaults to the repository root (same directory as this script).
#
# Requirements:
#   pip install papermill
#   GPU instance with requirements_transformers.txt installed
#
# Each notebook is executed in-place; outputs are saved to *_executed.ipynb
# in the same directory as the source notebook. If a notebook fails, the
# script logs the error and continues with the remaining notebooks.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${1:-$REPO_ROOT}"

ENGLISH_CSV="$DATA_DIR/english.csv"
FILIPINO_CSV="$DATA_DIR/filipino.csv"

if [[ ! -f "$ENGLISH_CSV" ]]; then
  echo "ERROR: english.csv not found at $ENGLISH_CSV" >&2
  exit 1
fi
if [[ ! -f "$FILIPINO_CSV" ]]; then
  echo "ERROR: filipino.csv not found at $FILIPINO_CSV" >&2
  exit 1
fi

# Ordered list: baseline first, then monolingual (English/Filipino), then bilingual
NOTEBOOKS=(
  "Baseline/TF_IDF_and_LR.ipynb"

  "BERT/bert_english/bert_english.ipynb"
  "BERT/bert_tagalog/bert_tagalog.ipynb"
  "BERT/bert_bilingual/bert_bilingual.ipynb"

  "NeoBERT/neobert_english/neobert_english.ipynb"
  "NeoBERT/neobert_tagalog/neobert_tagalog.ipynb"
  "NeoBERT/neobert_bilingual/neobert_bilingual.ipynb"

  "Roberta_Tagalog/Roberta_Tagalog_english/RoBERTa_Tagalog_english.ipynb"
  "Roberta_Tagalog/Roberta_Tagalog_tagalog/RoBERTa_Tagalog_tagalog.ipynb"
  "Roberta_Tagalog/Roberta_Tagalog_bilingual/RoBERTa_Tagalog_bilingual.ipynb"

  "XLM_RoBERTa/XLM_RoBERTa_english/XLM_RoBERTa_english.ipynb"
  "XLM_RoBERTa/XLM_RoBERTa_tagalog/XLM_RoBERTa_tagalog.ipynb"
  "XLM_RoBERTa/XLM_RoBERTa_bilingual/XLM_RoBERTa_bilingual.ipynb"
)

FAILED=()
PASSED=()

for NB_REL in "${NOTEBOOKS[@]}"; do
  NB_PATH="$REPO_ROOT/$NB_REL"
  NB_DIR="$(dirname "$NB_PATH")"
  NB_BASE="$(basename "$NB_PATH" .ipynb)"
  OUTPUT_NB="$NB_DIR/${NB_BASE}_executed.ipynb"

  echo ""
  echo "=========================================="
  echo "Running: $NB_REL"
  echo "=========================================="

  # Copy data files into the notebook's directory so OUTPUT_DIR="." works
  cp "$ENGLISH_CSV" "$NB_DIR/english.csv"
  cp "$FILIPINO_CSV" "$NB_DIR/filipino.csv"

  # cd into the notebook's directory so the kernel's working directory matches
  # OUTPUT_DIR="." — all pd.read_csv / file writes resolve relative to NB_DIR.
  if (cd "$NB_DIR" && papermill "$NB_PATH" "$OUTPUT_NB" \
      --execution-timeout 86400 \
      --log-output \
      --kernel "${KERNEL:-python3}") 2>&1; then
    echo "PASSED: $NB_REL"
    PASSED+=("$NB_REL")
  else
    echo "FAILED: $NB_REL (see ${OUTPUT_NB} for cell-level traceback)"
    FAILED+=("$NB_REL")
  fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Passed (${#PASSED[@]}):"
for nb in "${PASSED[@]}"; do echo "  OK  $nb"; done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "Failed (${#FAILED[@]}):"
  for nb in "${FAILED[@]}"; do echo "  FAIL $nb"; done
  exit 1
fi

echo "All notebooks completed successfully."
