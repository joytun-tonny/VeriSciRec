"""
Learned multimodal relational verifier (novelty item 1) -- training and evaluation.

Design decision, stated explicitly because it matters for how these results should be
read: the model is trained on SILVER labels (the rule-based verify()'s own verdicts
over the full 10,000-instance corpus, ~11.5k relational claims) and evaluated on the
GOLD labels (the 68-instance human-verified checkpoint, held out entirely from
training -- never used for fitting, only for the final comparison below). This is the
same silver-train/gold-evaluate pattern the rule-based tau was calibrated with, and it
is a real, not circular, test of whether the learned model does BETTER than the rule
it was trained to imitate: the model's features include information the rule ignores
(degree-normalized bibliographic coupling, abstract-level text similarity), so if it
generalizes past the rule's own blind spots (see counterfactual.py's direction-
insensitivity finding), gold performance should exceed the rule's, not just match it.
If it doesn't exceed the rule, that is also a real, reportable result, not a bug.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from graph_utils import CitationGraph, load_relational_claims, verify
from hard_negatives import build_tfidf_index
from verifier_features import build_dataset, FEATURE_NAMES


def load_gold_labels(verified_subset_path: Path) -> Dict[Tuple[str, str], int]:
    """
    Uses `human_type == 'relational'` -- the EXPERT-ADJUDICATED type, per the
    manuscript's Section 4.1 ("the relational type assigned to any genuine
    paper-paper relationship... independently of the outcome"). Confirmed against
    the manuscript's own reported count: human_type=='relational' gives exactly 215
    on this 303-claim checkpoint, matching Section 4.1 exactly.

    NOT `relational.is_relational` -- that's the pre-annotation AUTO-HEURISTIC flag
    used to route claims to annotators in the first place (178 on this checkpoint,
    not 215); human adjudication overrides it, and this session's first version of
    this loader used the wrong field, silently working from the wrong 178-claim
    population until checked against the manuscript's own reported number.

    Of the 215 expert-confirmed relational claims, only 72 have both paper_a and
    paper_b resolved to real IDs -- the majority don't, which is itself a real,
    reportable entity-resolution gap (the manuscript's Limitations section already
    flags relation extraction as lexical/title-matching-based, not an oracle; this
    is the concrete size of that gap on the gold checkpoint). This function can only
    score claims with both endpoints resolved, so n here is smaller than the
    manuscript's reported n=149 for the "full relational gold slice" (Table 5) --
    that number evidently uses a different, more permissive completeness criterion
    than "both endpoints resolved to real paper IDs," which this scoped session did
    not fully reverse-engineer. Report n=72 as what it is: the entity-complete
    subset, not a reproduction of Table 5's n=149.
    """
    labels = {}
    with open(verified_subset_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            iid = rec["id"]
            for lbl in rec.get("hallucination_labels", []):
                if lbl.get("human_type") != "relational":
                    continue
                rel = lbl.get("relational") or {}
                if not (rel.get("paper_a") and rel.get("paper_b")):
                    continue  # entity resolution incomplete for this claim; can't score it
                hl = lbl.get("human_label")
                if hl not in ("supported", "unsupported"):
                    continue
                labels[(iid, lbl["claim_id"])] = 1 if hl == "supported" else 0
    return labels


def load_silver_labels(cite_results_path: Path) -> Dict[Tuple[str, str], int]:
    labels = {}
    with open(cite_results_path, encoding="utf-8") as f:
        for r in json.load(f):
            if r["verdict"] in ("supported", "unsupported"):
                labels[(r["id"], r["claim_id"])] = 1 if r["verdict"] == "supported" else 0
    return labels


def _prf(yt: np.ndarray, yp: np.ndarray):
    """Precision/recall/F1 for the positive class. Shared by bootstrap_prf1 (single-
    method CI) and paired_bootstrap_diff (paired two-method comparison) below, so
    the two never drift apart on how a tie/zero-division case is handled."""
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def bootstrap_prf1(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Precision/recall/F1 for the positive class, with bootstrap 95% CIs."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    point = _prf(y_true, y_pred)
    boots = np.zeros((n_boot, 3))
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = _prf(y_true[idx], y_pred[idx])
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return {
        "precision": point[0], "recall": point[1], "f1": point[2],
        "precision_ci": [float(lo[0]), float(hi[0])],
        "recall_ci": [float(lo[1]), float(hi[1])],
        "f1_ci": [float(lo[2]), float(hi[2])],
        "n": n,
    }


def paired_bootstrap_diff(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray,
                           n_boot: int = 2000, seed: int = 0) -> dict:
    """
    Paired bootstrap test for whether method A differs from method B on precision/
    recall/F1, when both are scored on the SAME claims (e.g. rule vs. logistic
    regression vs. fusion MLP, all evaluated on the same gold set).

    Why this is more powerful than comparing two bootstrap_prf1() CIs side by side:
    that comparison independently resamples for each method, so the CI overlap test
    also has to absorb the sampling noise the two methods actually SHARE (the same
    claims are hard or easy for both), which inflates apparent overlap and makes two
    methods look statistically indistinguishable even when one is consistently
    better claim-for-claim. Resampling ONE set of claim indices per bootstrap
    iteration and scoring BOTH methods on that same resample (mirrored below) lets
    that shared variance cancel out of the difference -- the same reason a paired
    t-test has more power than an unpaired one on the same underlying data.

    Returns point-estimate A-minus-B differences with 95% CIs, plus a boolean
    `*_significant` flag per metric (True iff the CI excludes 0). At small gold-set
    sizes this can still come back non-significant for a real difference -- it's a
    strictly more sensitive test than the two-CI comparison, not a guarantee of
    finding significance where there isn't enough data to support it.
    """
    n = len(y_true)
    point_a, point_b = _prf(y_true, y_pred_a), _prf(y_true, y_pred_b)
    point_diff = tuple(a - b for a, b in zip(point_a, point_b))

    rng = np.random.RandomState(seed)
    diffs = np.zeros((n_boot, 3))
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)  # ONE resample, scored under BOTH methods -- the paired part
        pa, pb = _prf(y_true[idx], y_pred_a[idx]), _prf(y_true[idx], y_pred_b[idx])
        diffs[i] = [x - y for x, y in zip(pa, pb)]

    lo = np.percentile(diffs, 2.5, axis=0)
    hi = np.percentile(diffs, 97.5, axis=0)

    result = {"n": n}
    for i, name in enumerate(("precision", "recall", "f1")):
        result[f"{name}_diff"] = float(point_diff[i])
        result[f"{name}_diff_ci"] = [float(lo[i]), float(hi[i])]
        result[f"{name}_significant"] = bool(lo[i] > 0 or hi[i] < 0)
    return result


def main():
    root = Path(__file__).resolve().parents[2]
    refs_cache = root / "benchmarks/final/relational/citecheck_refs_cache.json"
    cite_results = root / "benchmarks/final/relational/citecheck_results.json"
    bench = root / "benchmarks/final/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"
    bench_verified = root / "benchmarks/final/nb2-output/scireqa_human_verified_subset.jsonl"

    print("Loading graph and claims...")
    graph = CitationGraph.load(refs_cache)
    all_claims = load_relational_claims(bench, cite_results)

    print("Building TF-IDF index over benchmark abstracts...")
    paper_abstracts = {}
    with open(bench, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for gt in rec.get("ground_truth_papers") or []:
                pid, ab = gt.get("paper_id"), gt.get("abstract")
                if pid and ab:
                    paper_abstracts[pid] = ab
    tfidf_vectors = build_tfidf_index(paper_abstracts)
    print(f"  {len(paper_abstracts)} papers with abstracts")

    silver_labels = load_silver_labels(cite_results)
    gold_labels = load_gold_labels(bench_verified)
    print(f"Silver-labeled claims: {len(silver_labels)} | Gold-labeled relational claims: {len(gold_labels)}")

    # Exclude any claim whose (instance_id, claim_id) is in the gold set from silver
    # training data, so gold stays a genuinely held-out test set even though both
    # sources ultimately derive from the same underlying benchmark instances.
    silver_labels_clean = {k: v for k, v in silver_labels.items() if k not in gold_labels}
    print(f"Silver claims after removing gold overlap: {len(silver_labels_clean)}")

    X_train, y_train, _ = build_dataset(all_claims, graph, tfidf_vectors, silver_labels_clean)
    X_gold, y_gold, gold_meta = build_dataset(all_claims, graph, tfidf_vectors, gold_labels)
    print(f"Training set: X={X_train.shape} | Gold eval set: X={X_gold.shape}")
    print(f"Gold label balance: {int(y_gold.sum())} supported / {len(y_gold)} total")

    results = {"feature_names": FEATURE_NAMES, "n_train": int(len(y_train)), "n_gold": int(len(y_gold))}

    # -- Baseline: the rule itself, on the gold set --
    rule_preds = np.array([
        1 if verify(m["paper_a"], m["relation"], m["paper_b"], graph) == "supported" else 0
        for m in gold_meta
    ])
    results["rule_based"] = bootstrap_prf1(y_gold, rule_preds)
    print("\nRule-based (baseline):", {k: v for k, v in results["rule_based"].items() if "ci" not in k})

    # -- Logistic regression --
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    Xtr_s, Xgold_s = scaler.transform(X_train), scaler.transform(X_gold)

    logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
    logreg.fit(Xtr_s, y_train)
    logreg_preds = logreg.predict(Xgold_s)
    results["logistic_regression"] = bootstrap_prf1(y_gold, logreg_preds)
    print("Logistic regression:", {k: v for k, v in results["logistic_regression"].items() if "ci" not in k})

    # -- Small MLP fusion classifier (matches the proposed h -> softmax(Wh+b) shape) --
    import torch
    import torch.nn as nn

    torch.manual_seed(0)

    class FusionMLP(nn.Module):
        def __init__(self, in_dim, hidden=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Linear(hidden // 2, 2),
            )

        def forward(self, x):
            return self.net(x)

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32)
    ytr_t = torch.tensor(y_train, dtype=torch.long)
    Xgold_t = torch.tensor(Xgold_s, dtype=torch.float32)

    # class-balanced loss weight, since silver "supported" heavily outnumbers "unsupported"
    n_pos, n_neg = int(y_train.sum()), int((1 - y_train).sum())
    class_weight = torch.tensor([n_pos / max(1, n_neg + n_pos), n_neg / max(1, n_neg + n_pos)], dtype=torch.float32)

    model = FusionMLP(X_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    n_epochs = 60
    batch_size = 256
    n_samples = Xtr_t.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch}: train loss {total_loss / n_samples:.4f}")

    model.eval()
    with torch.no_grad():
        mlp_preds = model(Xgold_t).argmax(dim=1).numpy()
    results["fusion_mlp"] = bootstrap_prf1(y_gold, mlp_preds)
    print("Fusion MLP:", {k: v for k, v in results["fusion_mlp"].items() if "ci" not in k})

    # -- Paired comparisons: more powerful than eyeballing CI overlap above, since it's
    # the same gold claims scored under every method (see paired_bootstrap_diff docstring) --
    print("\nPaired bootstrap comparisons (positive = first method ahead):")
    pairs = [("rule_vs_logreg", rule_preds, logreg_preds),
             ("rule_vs_fusion_mlp", rule_preds, mlp_preds),
             ("logreg_vs_fusion_mlp", logreg_preds, mlp_preds)]
    for name, preds_a, preds_b in pairs:
        cmp = paired_bootstrap_diff(y_gold, preds_a, preds_b)
        results[f"paired_{name}"] = cmp
        sig = " *SIGNIFICANT*" if cmp["f1_significant"] else ""
        print(f"  {name}: F1 diff={cmp['f1_diff']:+.3f} CI=[{cmp['f1_diff_ci'][0]:+.3f}, "
              f"{cmp['f1_diff_ci'][1]:+.3f}]{sig}")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "verifier_model_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), out_dir / "fusion_mlp.pt")
    print(f"\nSaved results to {out_dir / 'verifier_model_results.json'}")
    print(f"Saved model weights to {out_dir / 'fusion_mlp.pt'}")


if __name__ == "__main__":
    main()
