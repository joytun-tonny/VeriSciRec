"""
Verification-aware rationale revision (novelty item 8).

Three actions the original design proposal asks to compare:
  1. candidate_exclusion       -- drop the whole rationale/candidate if it contains
                                   any unsupported relational claim. This is what the
                                   deployed pipeline already does (Section 3.3's
                                   verified output, Eq. 22); it's the reference point,
                                   not something new to build.
  2. remove_unsupported_sentence -- keep the rationale, delete only the flagged
                                   sentence(s). Fully local/deterministic (no LLM
                                   call), same sentence-pruning logic already used to
                                   build DPO's "chosen" side -- reused here, not
                                   reimplemented, so both stay consistent.
  3. evidence_constrained_regeneration -- ask the generation model to rewrite the
                                   rationale grounded only in the verified claims.
                                   Needs a real LLM call (OPENROUTER_API_KEY), which
                                   is not available in this local session -- built as
                                   ready-to-run, not executed here. See the __main__
                                   block's second half.

SCOPE NOTE, read before comparing these numbers to Table 6: Table 6's "10 -> 7.7"
yield figure comes from the DEPLOYED PIPELINE's per-candidate live generation (one
rationale per candidate paper, 2,040 real calls). SciRecBench's released JSONL has a
different granularity -- one combined rationale per instance, covering the whole
recommended set. What's measured here is the rationale-level analogue: of the
instances an exclusion policy would have discarded entirely, how many survive with
at least some content after sentence-level pruning instead? That's the right
question for this data; it is not a literal re-run of Table 6's per-candidate number.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import List

from graph_utils import CitationGraph, verify

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT.split(str(text).strip()) if s.strip()]


def flagged_relational_sentences(instance: dict, graph: CitationGraph) -> set:
    """Sentence texts containing a relational claim that verify() marks unsupported."""
    flagged = set()
    for lbl in instance.get("hallucination_labels", []):
        rel = lbl.get("relational")
        if not rel or not rel.get("is_relational"):
            continue
        a, rho, b = rel.get("paper_a"), rel.get("relation"), rel.get("paper_b")
        if not (a and rho and b):
            continue
        if verify(a, rho, b, graph) == "unsupported":
            flagged.add((lbl.get("claim_text") or "").strip())
    return flagged


def revise_by_sentence_removal(rationale: str, flagged: set) -> str:
    kept = [s for s in split_sentences(rationale) if s not in flagged]
    return " ".join(kept)


def evaluate_corpus(instances: List[dict], graph: CitationGraph) -> dict:
    n = len(instances)
    n_would_exclude = 0          # exclusion policy: instance discarded entirely
    n_pruned_survives = 0        # sentence-removal: rationale still non-empty after pruning
    n_pruned_empty = 0           # sentence-removal removed EVERYTHING (still a zero-yield case)
    n_no_flagged = 0             # nothing to revise -- already clean
    orig_lengths, pruned_lengths = [], []

    for inst in instances:
        rationale = (inst.get("reference_rationale") or "").strip()
        if not rationale:
            continue
        flagged = flagged_relational_sentences(inst, graph)
        if not flagged:
            n_no_flagged += 1
            continue

        n_would_exclude += 1
        pruned = revise_by_sentence_removal(rationale, flagged)
        orig_lengths.append(len(split_sentences(rationale)))
        pruned_lengths.append(len(split_sentences(pruned)))
        if pruned.strip():
            n_pruned_survives += 1
        else:
            n_pruned_empty += 1

    return {
        "n_instances": n,
        "n_no_flagged_claims": n_no_flagged,
        "n_would_be_excluded": n_would_exclude,
        "exclusion_discard_rate": n_would_exclude / n if n else None,
        "sentence_removal_survives": n_pruned_survives,
        "sentence_removal_still_empty": n_pruned_empty,
        "sentence_removal_recovery_rate": n_pruned_survives / n_would_exclude if n_would_exclude else None,
        "avg_sentences_before": sum(orig_lengths) / len(orig_lengths) if orig_lengths else None,
        "avg_sentences_after": sum(pruned_lengths) / len(pruned_lengths) if pruned_lengths else None,
    }


# -- Strategy 3 template: needs a real LLM call, not executed in this local session --

def build_regeneration_prompt(rationale: str, flagged: set, supported_claims: List[str]) -> str:
    removed = "\n".join(f"- {s}" for s in flagged)
    kept = "\n".join(f"- {s}" for s in supported_claims) or "(none listed)"
    return (
        "Rewrite the following recommendation rationale in 2-4 fluent sentences. "
        "Do NOT include any of the REMOVED claims below, and do not introduce any new "
        "relationship between papers that is not already stated in the KEPT claims. "
        "It is fine for the rewrite to be shorter than the original.\n\n"
        f"ORIGINAL RATIONALE:\n{rationale}\n\n"
        f"REMOVED (unsupported) CLAIMS:\n{removed}\n\n"
        f"KEPT (supported) CLAIMS:\n{kept}\n"
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    refs_cache = root / "benchmarks/final/relational/citecheck_refs_cache.json"
    bench_full = root / "benchmarks/final/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"
    bench_verified = root / "benchmarks/final/nb2-output/scireqa_human_verified_subset.jsonl"

    graph = CitationGraph.load(refs_cache)

    def load_jsonl(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    full_corpus = load_jsonl(bench_full)
    verified_68 = load_jsonl(bench_verified)
    print(f"Full corpus: {len(full_corpus)} instances | Human-verified checkpoint: {len(verified_68)} instances")

    results = {
        "full_corpus_10000": evaluate_corpus(full_corpus, graph),
        "human_verified_68": evaluate_corpus(verified_68, graph),
    }

    for name, r in results.items():
        print(f"\n--- {name} ---")
        for k, v in r.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.1%}" if "rate" in k else f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "rationale_revision_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'rationale_revision_results.json'}")
