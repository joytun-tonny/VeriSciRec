#!/usr/bin/env bash
# Build the compressed SciRecBench data bundle that is attached to a GitHub Release
# (files are too large for git: GitHub rejects files > 100 MB).
#
# Usage (from the repository root):
#     bash scripts/make_release.sh
#     gh release create v1.1 release/* --title "SciRecBench v1.1" --notes-file DATA.md
#
# Output goes to ./release/ (git-ignored). Nothing in the working tree is modified.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
B="$ROOT/benchmarks/final"
OUT="$ROOT/release"
mkdir -p "$OUT"

FILES=(
  # Relational benchmark (canonical release, with expert labels merged in)
  "$B/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"
  # Per-domain relational splits
  "$B/relational/scireqa_arxiv_relational_v1.1.jsonl"
  "$B/relational/scireqa_acl_relational_v1.1.jsonl"
  "$B/relational/scireqa_pubmed_relational_v1.1.jsonl"
  # Atomic-claim benchmark (v1.1, after label diagnosis)
  "$B/after_diagnosis/scireqa_full_v1.1.FINAL.jsonl"
  "$B/after_diagnosis/scireqa_full_with_negatives_v1.1.FINAL.jsonl"
  "$B/after_diagnosis/scireqa_hallucination_eval_v1.1.FINAL.jsonl"
  # Cached citation-graph data used by the VeriSciRec verifier
  "$B/relational/citecheck_refs_cache.json"
  # Generated rationales (LLM outputs + prompts checkpoint)
  "$B/relational/rationales_checkpoint_relational_v1_1.json"
)

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then echo "missing: $f" >&2; exit 1; fi
  name="$(basename "$f").gz"
  echo "compressing $(basename "$f") ..."
  gzip -9 -c "$f" > "$OUT/$name"
done

( cd "$OUT" && sha256sum *.gz > SHA256SUMS.txt )
echo
ls -lh "$OUT"
echo "GitHub allows release assets up to 2 GB each."
