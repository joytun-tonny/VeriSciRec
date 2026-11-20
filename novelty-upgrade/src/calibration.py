"""
Calibrated abstention via split-conformal prediction (novelty item 3), layered on top
of the learned verifier from verifier_model.py.

Method: LAC (least ambiguous set-valued classifier) split-conformal, chosen over the
fancier APS (adaptive prediction sets) construction because this is binary
classification -- LAC's simpler score s(x, y) = 1 - p_hat(y | x) is enough to get a
finite-sample marginal coverage guarantee here, and APS's main advantage (better
handling of the "which OTHER classes to include" question) doesn't apply with only
two classes. Prediction set C(x) = {y : s(x, y) <= qhat}; a claim is "abstained on"
when |C(x)| != 1 (either both labels survive -- genuinely ambiguous -- or neither
does, which for LAC only happens if qhat < 0, i.e. never, but the check is kept for
safety).

TWO separate calibration runs are reported, deliberately, because they answer
different questions and only one of them is methodologically airtight:

  1. GOLD-calibrated (in-distribution): gold set split into calib/test halves. This
     is the textbook conformal setup -- calibration and test data are exchangeable by
     construction, so the marginal coverage guarantee actually holds. The catch: gold
     has ~70 usable claims (see verifier_model.py's load_gold_labels docstring), so a
     50/50 split leaves ~35 calib / ~35 test. The guarantee is real but the interval
     around "did we hit 90% coverage" is wide at that n -- reported honestly via
     bootstrap CI, not hidden.

  2. SILVER-calibrated, GOLD-evaluated (distribution-shift test): calibrate qhat on a
     held-out slice of SILVER labels (the rule's own verdicts -- there's a lot of this
     data, so qhat itself is well-estimated), then check empirical coverage on the
     FULL gold set. This is NOT a valid conformal guarantee in the textbook sense --
     silver and gold are not exchangeable, since silver labels are literally what the
     rule-based verifier said, not independent human judgments. It is included anyway
     because the gap between nominal coverage (90%) and what's actually achieved on
     gold IS the finding: if silver-calibration transfers, that's a practically useful
     shortcut (calibrate cheaply, deploy against reality); if it doesn't, that is a
     genuine, reportable limitation of using the rule's own output as a calibration
     surrogate, and it's exactly the kind of gap that argues for growing the gold set
     rather than papering over it.

Both runs use the same underlying model (logistic regression on the 16-feature
verifier_features.py representation, trained on silver labels with the gold claims
excluded) so the two calibration strategies are compared on equal footing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def nonconformity_scores(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAC score: s(x, y) = 1 - p_hat(y | x). Lower means the model was more
    confident and more correct about that particular label."""
    return 1.0 - probs[np.arange(len(y)), y]


def calibrate_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of the calibration non-conformity
    scores, computed by explicit sorting/indexing rather than np.quantile's
    interpolation argument, whose name changed across numpy versions
    (interpolation= vs method=) -- this sidesteps that entirely."""
    n = len(cal_scores)
    sorted_scores = np.sort(cal_scores)
    idx = int(np.ceil((n + 1) * (1 - alpha))) - 1
    idx = min(max(idx, 0), n - 1)
    return float(sorted_scores[idx])


def predict_sets(probs: np.ndarray, qhat: float) -> List[Tuple[int, ...]]:
    """Returns, for each row of probs, the tuple of labels y in {0, 1} whose
    non-conformity score is within qhat -- the conformal prediction set."""
    sets = []
    for p in probs:
        s = tuple(y for y in (0, 1) if (1.0 - p[y]) <= qhat)
        sets.append(s)
    return sets


def bootstrap_metric(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Generic bootstrap mean + 95% CI over a 0/1 (or float) array, used for coverage
    and selective-accuracy CIs below."""
    rng = np.random.RandomState(seed)
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = values[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def evaluate_conformal(sets: List[Tuple[int, ...]], y_true: np.ndarray) -> dict:
    n = len(y_true)
    covered = np.array([1.0 if y_true[i] in sets[i] else 0.0 for i in range(n)])
    set_sizes = np.array([len(s) for s in sets])
    singleton_idx = np.where(set_sizes == 1)[0]
    ambiguous_idx = np.where(set_sizes != 1)[0]

    cov_mean, cov_lo, cov_hi = bootstrap_metric(covered)

    selective_accuracy = None
    sel_lo = sel_hi = None
    if len(singleton_idx) > 0:
        correct = np.array([
            1.0 if sets[i][0] == y_true[i] else 0.0 for i in singleton_idx
        ])
        selective_accuracy, sel_lo, sel_hi = bootstrap_metric(correct)

    return {
        "n": n,
        "coverage": cov_mean, "coverage_ci": [cov_lo, cov_hi],
        "avg_set_size": float(set_sizes.mean()),
        "abstention_rate": float(len(ambiguous_idx) / n),
        "n_singleton": int(len(singleton_idx)),
        "selective_accuracy": selective_accuracy,
        "selective_accuracy_ci": [sel_lo, sel_hi] if selective_accuracy is not None else None,
    }


def split_indices(n: int, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    cut = int(round(n * frac))
    return idx[:cut], idx[cut:]


def run_calibration_suite(model, scaler, X_silver_val, y_silver_val, X_gold, y_gold,
                           alphas=(0.1, 0.2), seed: int = 0) -> dict:
    """Runs both calibration strategies (silver->gold transfer, and gold-internal
    split) at each alpha in `alphas`. `model` must expose predict_proba; `scaler`
    is applied to all X before scoring, matching how the model was trained."""
    probs_silver_val = model.predict_proba(scaler.transform(X_silver_val))
    probs_gold = model.predict_proba(scaler.transform(X_gold))

    results = {"n_silver_val": int(len(y_silver_val)), "n_gold": int(len(y_gold)), "alphas": {}}

    for alpha in alphas:
        alpha_key = f"alpha_{alpha}"
        target_coverage = 1 - alpha
        block = {"target_coverage": target_coverage}

        # -- Strategy 1: silver-calibrated, gold-evaluated (distribution-shift test) --
        scores_silver_val = nonconformity_scores(probs_silver_val, y_silver_val)
        qhat_silver = calibrate_threshold(scores_silver_val, alpha)
        sets_gold_via_silver = predict_sets(probs_gold, qhat_silver)
        block["silver_calibrated_on_gold"] = {
            "qhat": qhat_silver,
            **evaluate_conformal(sets_gold_via_silver, y_gold),
        }

        # -- Strategy 2: gold-internal split (valid conformal guarantee, tiny n) --
        calib_idx, test_idx = split_indices(len(y_gold), frac=0.5, seed=seed)
        if len(calib_idx) >= 5 and len(test_idx) >= 5:
            scores_gold_calib = nonconformity_scores(probs_gold[calib_idx], y_gold[calib_idx])
            qhat_gold = calibrate_threshold(scores_gold_calib, alpha)
            sets_gold_test = predict_sets(probs_gold[test_idx], qhat_gold)
            block["gold_internal_split"] = {
                "qhat": qhat_gold,
                "n_calib": int(len(calib_idx)),
                **evaluate_conformal(sets_gold_test, y_gold[test_idx]),
            }
        else:
            block["gold_internal_split"] = {"note": "too few gold claims for a 50/50 split"}

        results["alphas"][alpha_key] = block

    return results


def main():
    root = Path(__file__).resolve().parents[2]
    refs_cache = root / "benchmarks/final/relational/citecheck_refs_cache.json"
    cite_results = root / "benchmarks/final/relational/citecheck_results.json"
    bench = root / "benchmarks/final/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"
    bench_verified = root / "benchmarks/final/nb2-output/scireqa_human_verified_subset.jsonl"

    from graph_utils import CitationGraph, load_relational_claims
    from hard_negatives import build_tfidf_index
    from verifier_features import build_dataset
    import verifier_model as vm
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

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

    silver = vm.load_silver_labels(cite_results)
    gold = vm.load_gold_labels(bench_verified)
    silver_clean = {k: v for k, v in silver.items() if k not in gold}

    X_silver, y_silver, meta_silver = build_dataset(all_claims, graph, tfidf_vectors, silver_clean)
    X_gold, y_gold, meta_gold = build_dataset(all_claims, graph, tfidf_vectors, gold)
    print(f"Silver: {X_silver.shape} | Gold: {X_gold.shape}")

    # Hold out a slice of silver purely for calibration -- never used to fit the model,
    # matching the same silver-train / silver-val separation principle as
    # silver-train / gold-eval elsewhere in this project.
    train_idx, calib_idx = split_indices(len(y_silver), frac=0.85, seed=0)
    X_tr, y_tr = X_silver[train_idx], y_silver[train_idx]
    X_cal, y_cal = X_silver[calib_idx], y_silver[calib_idx]
    print(f"Silver train: {len(y_tr)} | Silver calibration slice: {len(y_cal)}")

    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(scaler.transform(X_tr), y_tr)

    print("\nRunning conformal calibration suite (alpha=0.1 and alpha=0.2)...")
    results = run_calibration_suite(model, scaler, X_cal, y_cal, X_gold, y_gold, alphas=(0.1, 0.2), seed=0)

    for alpha_key, block in results["alphas"].items():
        print(f"\n--- {alpha_key} (target coverage {block['target_coverage']:.0%}) ---")
        s = block["silver_calibrated_on_gold"]
        print(f"  silver-calibrated -> gold: coverage={s['coverage']:.3f} "
              f"[{s['coverage_ci'][0]:.3f},{s['coverage_ci'][1]:.3f}] "
              f"abstention_rate={s['abstention_rate']:.3f} avg_set_size={s['avg_set_size']:.3f}")
        g = block["gold_internal_split"]
        if "coverage" in g:
            print(f"  gold-internal split:      coverage={g['coverage']:.3f} "
                  f"[{g['coverage_ci'][0]:.3f},{g['coverage_ci'][1]:.3f}] "
                  f"abstention_rate={g['abstention_rate']:.3f} avg_set_size={g['avg_set_size']:.3f} "
                  f"(n_calib={g['n_calib']}, n_test={g['n']})")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "calibration_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_dir / 'calibration_results.json'}")


if __name__ == "__main__":
    main()
