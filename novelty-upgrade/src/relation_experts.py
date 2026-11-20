"""
Relation-specific expert routing (novelty item 2).

SCOPED DOWN from the originally proposed full 5-way mixture-of-experts, and why: the
relation taxonomy is badly imbalanced in the actual corpus --
builds_on=6,183, shares_topic=2,857, compares=2,394, uses_method_of unknown but
folded into "dependency," supersedes=83. Eighty-three examples corpus-wide is not
enough to train a DEDICATED learned expert for `supersedes` without serious
overfitting risk, and the same concern applies less severely to any single
fine-grained relation type once split five ways.

What's actually built here: a 2-way router (dependency vs. non-dependency), which is
the one split the rule-based decision rule ITSELF already treats differently (Eq. 14
literally branches on `rho in D`), so it's the split with the most reason to expect
a real difference in what evidence matters -- and both sides have thousands of
examples (dependency: builds_on+uses_method_of+supersedes; non-dependency:
compares+shares_topic). The finer 5-way split from the original design proposal, and the
router ALSO re-estimating whether the extracted relation label itself is correct
(their P(rho|c) term), are documented here as a follow-up once more gold data
narrows the confidence intervals enough to tell whether the extra granularity earns
its complexity -- building it now, on this little gold data, would not be
distinguishable from noise any more clearly than the single pooled model was.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from graph_utils import CitationGraph, DEPENDENCY_RELATIONS, load_relational_claims, verify
from verifier_features import build_dataset, FEATURE_NAMES
from verifier_model import bootstrap_prf1, load_gold_labels, load_silver_labels, paired_bootstrap_diff


def route(rho: str) -> str:
    return "dependency" if rho in DEPENDENCY_RELATIONS else "non_dependency"


def split_by_route(X: np.ndarray, y: np.ndarray, meta: List[dict]):
    """Splits an already-built (X, y, meta) dataset into per-route subsets."""
    idx_dep = [i for i, m in enumerate(meta) if route(m["relation"]) == "dependency"]
    idx_nondep = [i for i, m in enumerate(meta) if route(m["relation"]) == "non_dependency"]
    return {
        "dependency": (X[idx_dep], y[idx_dep], [meta[i] for i in idx_dep]),
        "non_dependency": (X[idx_nondep], y[idx_nondep], [meta[i] for i in idx_nondep]),
    }


def train_routed_experts(X_train, y_train, meta_train):
    """One LogisticRegression per route. Logistic regression, not the MLP, deliberately --
    each expert sees roughly half the pooled training data, and a simpler model is the
    safer choice when per-expert sample size drops (same reasoning as scoping out the
    5-way split above)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    splits = split_by_route(X_train, y_train, meta_train)
    experts = {}
    for route_name, (X_r, y_r, _) in splits.items():
        scaler = StandardScaler().fit(X_r)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_r), y_r)
        experts[route_name] = (clf, scaler)
        print(f"  trained {route_name} expert on {len(y_r)} claims "
              f"({int(y_r.sum())} supported / {len(y_r)})")
    return experts


def predict_routed(experts: dict, X: np.ndarray, meta: List[dict]) -> np.ndarray:
    preds = np.zeros(len(meta), dtype=int)
    for i, m in enumerate(meta):
        clf, scaler = experts[route(m["relation"])]
        preds[i] = clf.predict(scaler.transform(X[i:i + 1]))[0]
    return preds


def main():
    root = Path(__file__).resolve().parents[2]
    refs_cache = root / "benchmarks/final/relational/citecheck_refs_cache.json"
    cite_results = root / "benchmarks/final/relational/citecheck_results.json"
    bench = root / "benchmarks/final/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"
    bench_verified = root / "benchmarks/final/nb2-output/scireqa_human_verified_subset.jsonl"

    from hard_negatives import build_tfidf_index

    graph = CitationGraph.load(refs_cache)
    all_claims = load_relational_claims(bench, cite_results)

    paper_abstracts = {}
    with open(bench, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for gt in rec.get("ground_truth_papers") or []:
                pid, ab = gt.get("paper_id"), gt.get("abstract")
                if pid and ab:
                    paper_abstracts[pid] = ab
    tfidf_vectors = build_tfidf_index(paper_abstracts)

    silver = load_silver_labels(cite_results)
    gold = load_gold_labels(bench_verified)
    silver_clean = {k: v for k, v in silver.items() if k not in gold}

    X_train, y_train, meta_train = build_dataset(all_claims, graph, tfidf_vectors, silver_clean)
    X_gold, y_gold, meta_gold = build_dataset(all_claims, graph, tfidf_vectors, gold)
    print(f"Train: {X_train.shape} | Gold: {X_gold.shape}")

    print("\nTraining routed experts...")
    experts = train_routed_experts(X_train, y_train, meta_train)
    routed_preds = predict_routed(experts, X_gold, meta_gold)

    print("\nTraining single pooled model for comparison...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    pooled = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_train), y_train)
    pooled_preds = pooled.predict(scaler.transform(X_gold))

    rule_preds = np.array([
        1 if verify(m["paper_a"], m["relation"], m["paper_b"], graph) == "supported" else 0
        for m in meta_gold
    ])

    results = {
        "rule_based": bootstrap_prf1(y_gold, rule_preds),
        "pooled_logreg": bootstrap_prf1(y_gold, pooled_preds),
        "routed_2expert_logreg": bootstrap_prf1(y_gold, routed_preds),
    }
    for name, r in results.items():
        print(f"{name}: precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1']:.3f} "
              f"(F1 CI [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}])")

    # Paired comparison, overall: same gold claims scored under both, so this is a
    # more sensitive test than eyeballing the two CIs above for overlap (see
    # verifier_model.paired_bootstrap_diff's docstring).
    overall_cmp = paired_bootstrap_diff(y_gold, pooled_preds, routed_preds)
    results["paired_pooled_vs_routed"] = overall_cmp
    sig = " *SIGNIFICANT*" if overall_cmp["f1_significant"] else ""
    print(f"\nPaired pooled-vs-routed: F1 diff={overall_cmp['f1_diff']:+.3f} "
          f"CI=[{overall_cmp['f1_diff_ci'][0]:+.3f}, {overall_cmp['f1_diff_ci'][1]:+.3f}]{sig}")

    # per-route breakdown, since the whole point is whether routing helps one route
    # more than the other -- paired within each route for the same reason as above
    for route_name in ("dependency", "non_dependency"):
        idx = [i for i, m in enumerate(meta_gold) if route(m["relation"]) == route_name]
        if not idx:
            continue
        yt, yp_pooled, yp_routed = y_gold[idx], pooled_preds[idx], routed_preds[idx]
        r_pooled = bootstrap_prf1(yt, yp_pooled, n_boot=500)
        r_routed = bootstrap_prf1(yt, yp_routed, n_boot=500)
        route_cmp = paired_bootstrap_diff(yt, yp_pooled, yp_routed, n_boot=500)
        sig = " *SIGNIFICANT*" if route_cmp["f1_significant"] else ""
        print(f"\n[{route_name}, n={len(idx)}] pooled F1={r_pooled['f1']:.3f} | routed F1={r_routed['f1']:.3f} "
              f"| paired diff={route_cmp['f1_diff']:+.3f} "
              f"CI=[{route_cmp['f1_diff_ci'][0]:+.3f}, {route_cmp['f1_diff_ci'][1]:+.3f}]{sig}")
        results[f"{route_name}_pooled"] = r_pooled
        results[f"{route_name}_routed"] = r_routed
        results[f"paired_{route_name}_pooled_vs_routed"] = route_cmp

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "relation_experts_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'relation_experts_results.json'}")


if __name__ == "__main__":
    main()
