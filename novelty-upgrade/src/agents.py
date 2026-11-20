"""
Real agentic verification loop (novelty item 7).

The design concern, as scoped here: a chain of LLM calls that always
runs the same fixed sequence of steps in the same order, regardless of what any step
actually finds, is a PIPELINE, not an agent -- nothing in it decides what to do next
based on what it just observed. This module builds a small, fully offline,
deterministic decision loop (AgenticVerifier) that DOES have the properties usually
meant by "agentic":

  - Tool selection conditioned on uncertainty: the cheap tool (the rule-based
    verify()) always runs first; the direction-reversal probe only runs for
    dependency-relation claims the rule marked "supported" (the one case
    counterfactual.py found to be direction-fragile -- 19.1% flip rate -- vs. 0% for
    non-dependency relations, so probing non-dependency claims would burn a tool call
    for a decision that's already known not to move); the learned model and conformal
    calibration only run when something upstream was actually uncertain (fragile
    under the probe, or the rule itself said "unverifiable"). A fixed pipeline would
    run every tool on every claim; this loop visibly does not.
  - Self-correction via counterfactual probing: before finalizing a "supported"
    verdict, the loop actually re-verifies THIS SPECIFIC claim under a reversed
    citation edge and downgrades confidence (routes to escalation) if the verdict
    isn't robust to it, instead of only reporting fragility in an offline aggregate
    the way notebook 02 does.
  - A real stop/abstain decision: escalated claims go through calibration.py's
    conformal prediction set, and an ambiguous set (both labels survive) becomes an
    explicit "abstain, flag for human review" outcome -- not a fixed 0.5-probability
    cutoff that always answers.
  - A visible reasoning trace: every decision records which tools it called, what
    each observed, and why it branched the way it did (`AgentDecision.trace`) --
    inspectable evidence of the control flow actually taken per claim, not just a
    final label.

SCOPED to reuse only already-built, already-validated components from this session
(graph_utils.verify, counterfactual.perturb_reverse_direction, the learned verifier
from verifier_model.py, calibration.py's conformal machinery) -- no new LLM calls, no
new API cost, runs anywhere the rest of novelty-upgrade/ runs. Retrofitting this same
control-flow pattern into the existing recommendation pipeline's LLM-based agents (so
those steps also branch on intermediate confidence instead of always running a fixed
number of times) is the natural next step once this loop's own decisions have been
validated against gold -- not attempted blind against pipeline code this session
never re-read, per the "build fresh, don't patch the original notebooks" instruction
that scoped this whole folder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from graph_utils import CitationGraph, DEPENDENCY_RELATIONS, verify
from counterfactual import perturb_reverse_direction
from calibration import predict_sets


@dataclass
class AgentDecision:
    paper_a: str
    relation: str
    paper_b: str
    verdict: str  # "supported" | "unsupported" | "unverifiable" | "abstain"
    escalated: bool  # did it need the learned model + calibration, or did the cheap rule check settle it?
    fragile: bool  # did the direction-reversal probe flip the rule's own verdict?
    tools_called: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)


class AgenticVerifier:
    """A real decision loop over already-built verification components -- see module
    docstring for what makes this "agentic" rather than a single classifier call with
    an agent-shaped name."""

    def __init__(self, graph: CitationGraph, tfidf_vectors: Dict[str, dict],
                 learned_model=None, learned_scaler=None, conformal_qhat: Optional[float] = None):
        self.graph = graph
        self.tfidf_vectors = tfidf_vectors
        self.learned_model = learned_model
        self.learned_scaler = learned_scaler
        self.conformal_qhat = conformal_qhat

    def decide(self, a: str, rho: str, b: str) -> AgentDecision:
        trace: List[str] = []
        tools: List[str] = []

        if a not in self.graph or b not in self.graph:
            trace.append(f"paper not in graph (a_present={a in self.graph}, b_present={b in self.graph}) "
                          f"-> unverifiable, stopping without further tool calls")
            return AgentDecision(a, rho, b, "unverifiable", escalated=False, fragile=False,
                                  tools_called=tools, trace=trace)

        tools.append("rule_based_verify")
        rule_verdict = verify(a, rho, b, self.graph)
        trace.append(f"rule_based_verify(a, {rho}, b) -> {rule_verdict}")

        fragile = False
        if rho in DEPENDENCY_RELATIONS and rule_verdict == "supported":
            tools.append("counterfactual_reverse_direction_probe")
            g2, rho2 = perturb_reverse_direction(a, rho, b, self.graph)
            probe_verdict = verify(a, rho2, b, g2)
            fragile = probe_verdict != rule_verdict
            trace.append(
                f"direction-reversal probe (dependency relation, rule said supported) -> "
                f"{probe_verdict} ({'FRAGILE: flips under reversal' if fragile else 'stable under reversal'})"
            )
        else:
            trace.append(
                "skipped direction-reversal probe: non-dependency relations are empirically "
                "direction-insensitive (0% flip rate, see counterfactual.py) or the rule "
                "already said supported=False, so the probe would not change the decision"
            )

        needs_escalation = fragile or rule_verdict == "unverifiable"
        if not needs_escalation:
            trace.append(f"rule verdict stable and non-fragile -> final verdict {rule_verdict}, no escalation needed")
            return AgentDecision(a, rho, b, rule_verdict, escalated=False, fragile=fragile,
                                  tools_called=tools, trace=trace)

        if self.learned_model is None or self.conformal_qhat is None:
            trace.append("escalation triggered but no learned model / calibration wired in -- falling back to rule verdict")
            return AgentDecision(a, rho, b, rule_verdict, escalated=True, fragile=fragile,
                                  tools_called=tools, trace=trace)

        tools.append("learned_verifier")
        from verifier_features import claim_features
        x = claim_features(a, rho, b, self.graph, self.tfidf_vectors).reshape(1, -1)
        probs = self.learned_model.predict_proba(self.learned_scaler.transform(x))[0]
        trace.append(f"learned_verifier probability [unsupported={probs[0]:.3f}, supported={probs[1]:.3f}]")

        tools.append("conformal_abstention_check")
        pred_set = predict_sets(probs.reshape(1, -1), self.conformal_qhat)[0]
        trace.append(f"conformal prediction set at calibrated threshold: {pred_set}")

        if len(pred_set) != 1:
            trace.append("ambiguous prediction set -> abstain, flag for human review")
            return AgentDecision(a, rho, b, "abstain", escalated=True, fragile=fragile,
                                  tools_called=tools, trace=trace)

        final = "supported" if pred_set[0] == 1 else "unsupported"
        trace.append(f"singleton prediction set -> final verdict {final}")
        return AgentDecision(a, rho, b, final, escalated=True, fragile=fragile, tools_called=tools, trace=trace)


def evaluate_agent(agent: AgenticVerifier, claims_meta: List[dict], y_true: np.ndarray) -> dict:
    """claims_meta: the `meta` list from verifier_features.build_dataset (has
    paper_a/relation/paper_b), aligned 1:1 with y_true. Reports escalation/fragility/
    abstention rates plus precision/recall/F1 on whatever the agent actually
    answered (selective prediction -- abstained claims are excluded from the metric,
    same as calibration.py's selective_accuracy)."""
    from verifier_model import bootstrap_prf1

    decisions = [agent.decide(m["paper_a"], m["relation"], m["paper_b"]) for m in claims_meta]

    n = len(decisions)
    n_escalated = sum(d.escalated for d in decisions)
    n_fragile = sum(d.fragile for d in decisions)
    n_abstained = sum(d.verdict == "abstain" for d in decisions)

    answered_idx = [i for i, d in enumerate(decisions) if d.verdict in ("supported", "unsupported")]
    metrics = None
    if answered_idx:
        y_true_answered = y_true[answered_idx]
        y_pred_answered = np.array([1 if decisions[i].verdict == "supported" else 0 for i in answered_idx])
        metrics = bootstrap_prf1(y_true_answered, y_pred_answered)

    return {
        "n": n,
        "n_escalated": n_escalated, "escalation_rate": n_escalated / n if n else None,
        "n_fragile": n_fragile, "fragile_rate": n_fragile / n if n else None,
        "n_abstained": n_abstained, "abstention_rate": n_abstained / n if n else None,
        "n_answered": len(answered_idx),
        "metrics_on_answered": metrics,
        "sample_traces": [
            {"paper_a": d.paper_a, "relation": d.relation, "paper_b": d.paper_b,
             "verdict": d.verdict, "escalated": d.escalated, "fragile": d.fragile,
             "tools_called": d.tools_called, "trace": d.trace}
            for d in decisions[:5]
        ],
    }


# -- Qualitative case-study selection + LaTeX rendering, for the manuscript ---------
#
# Purpose: the manuscript's qualitative evidence (Appendix "Additional Qualitative
# Examples") currently only covers item 0 (the relational-hallucination blind spot).
# This gives item 7 (real agentic capabilities) the same treatment -- concrete, real
# worked examples a reviewer can read, not just an aggregate escalation-rate number.
# Nothing here is fabricated: it selects and formats real AgentDecision output on
# real gold claims. It produces NOTHING until run against real data on Kaggle --
# see build_09_agentic_verifier.py's notebook cell for where this gets called.

# Priority order for which (escalated, fragile, verdict) combination to showcase
# first -- ranked by how much each demonstrates behavior a FIXED pipeline could not
# produce (a static prompt chain cannot skip a step, cannot self-probe, cannot
# abstain), most distinctive first:
SHOWCASE_PRIORITY = [
    (True, True, "abstain"),       # self-correction found real fragility AND the system knew to stop
    (True, True, "supported"),     # self-correction found fragility but still resolved confidently
    (True, True, "unsupported"),
    (True, False, "abstain"),      # escalated for a different reason (rule said unverifiable), still abstained
    (True, False, "supported"),
    (True, False, "unsupported"),
    (False, False, "supported"),   # cheap path: rule alone was trusted, no wasted tool calls
    (False, False, "unsupported"),
    (False, False, "unverifiable"),  # cheapest path: stopped before any tool call (paper not in graph)
]


def select_showcase_examples(agent: AgenticVerifier, claims_meta: List[dict], y_true: np.ndarray,
                              max_total: int = 8) -> List[dict]:
    """One (decision, gold_label) pair per distinct behavior bucket actually observed
    in `claims_meta`, ordered by SHOWCASE_PRIORITY, capped at `max_total`. Buckets
    with zero real examples in the gold set are simply skipped -- never backfilled
    with anything synthetic."""
    decisions = [agent.decide(m["paper_a"], m["relation"], m["paper_b"]) for m in claims_meta]

    by_bucket: Dict[tuple, List[int]] = {}
    for i, d in enumerate(decisions):
        key = (d.escalated, d.fragile, d.verdict)
        by_bucket.setdefault(key, []).append(i)

    chosen = []
    for key in SHOWCASE_PRIORITY:
        if key not in by_bucket or len(chosen) >= max_total:
            continue
        i = by_bucket[key][0]  # first observed instance of this bucket, not cherry-picked further
        d, m = decisions[i], claims_meta[i]
        # Recompute the rule's own verdict directly, rather than parsing it back out of
        # trace text -- trace strings are for human reading, not for round-tripping data.
        rule_verdict = verify(m["paper_a"], d.relation, m["paper_b"], agent.graph)
        chosen.append({
            "instance_id": m["instance_id"], "claim_id": m["claim_id"],
            "relation": d.relation, "rule_verdict": rule_verdict,
            "fragile": d.fragile, "escalated": d.escalated, "verdict": d.verdict,
            "gold_label": "supported" if y_true[i] == 1 else "unsupported",
            "tools_called": d.tools_called, "trace": d.trace,
        })
    return chosen


def _latex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("%", r"\%").replace("#", r"\#").replace("&", r"\&"))


def render_showcase_latex(examples: List[dict], n_full_traces: int = 2) -> str:
    """Renders `examples` (from select_showcase_examples) as a ready-to-paste LaTeX
    fragment: a compact table*, in the same toprule/midrule/bottomrule \\footnotesize
    style as the manuscript's existing tab:qualexamples, plus `n_full_traces` fully
    spelled-out reasoning traces (the most distinctive buckets first) as itemized
    lists. Paste directly into the manuscript's "Additional Qualitative Examples"
    appendix in place of the \\TODO{} placeholder -- do not hand-edit the numbers."""
    lines = []
    lines.append(r"\begin{table*}[h]\centering\footnotesize")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"Relation & Rule & Fragile? & Escalated? & Agent verdict & Gold\\")
    lines.append(r"\midrule")
    for ex in examples:
        rel = _latex_escape(ex["relation"] or "")
        rule = _latex_escape(ex["rule_verdict"] or "")
        fragile = "yes" if ex["fragile"] else "no"
        escalated = "yes" if ex["escalated"] else "no"
        verdict = _latex_escape(ex["verdict"])
        gold = _latex_escape(ex["gold_label"])
        lines.append(f"\\textsf{{{rel}}} & {rule} & {fragile} & {escalated} & {verdict} & {gold}\\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Agentic verification loop behavior on a representative sample of gold claims, "
        r"one example per distinct (escalated, fragile, verdict) combination actually observed. "
        r"``Fragile'' marks dependency-relation claims whose rule-based verdict did not survive the "
        r"counterfactual direction-reversal probe; ``escalated'' marks claims routed past the rule to "
        r"the learned verifier and conformal calibration. Full reasoning traces for the most "
        r"distinctive cases follow.}"
    )
    lines.append(r"\label{tab:agentic_showcase}")
    lines.append(r"\end{table*}")
    lines.append("")

    for ex in examples[:n_full_traces]:
        lines.append(
            f"\\paragraph{{Example reasoning trace ({_latex_escape(ex['verdict'])}).}} "
            f"For a \\textsf{{{_latex_escape(ex['relation'] or '')}}} claim:"
        )
        lines.append(r"\begin{itemize}\footnotesize")
        for step in ex["trace"]:
            lines.append(f"  \\item {_latex_escape(step)}")
        lines.append(r"\end{itemize}")
        lines.append("")

    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parents[2]
    refs_cache = root / "benchmarks/final/relational/citecheck_refs_cache.json"
    cite_results = root / "benchmarks/final/relational/citecheck_results.json"
    bench = root / "benchmarks/final/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"
    bench_verified = root / "benchmarks/final/nb2-output/scireqa_human_verified_subset.jsonl"

    from graph_utils import load_relational_claims
    from hard_negatives import build_tfidf_index
    from verifier_features import build_dataset
    import verifier_model as vm
    from calibration import split_indices, calibrate_threshold, nonconformity_scores
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

    train_idx, calib_idx = split_indices(len(y_silver), frac=0.85, seed=0)
    X_tr, y_tr = X_silver[train_idx], y_silver[train_idx]
    X_cal, y_cal = X_silver[calib_idx], y_silver[calib_idx]

    scaler = StandardScaler().fit(X_tr)
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_tr), y_tr)

    probs_cal = model.predict_proba(scaler.transform(X_cal))
    scores_cal = nonconformity_scores(probs_cal, y_cal)
    qhat = calibrate_threshold(scores_cal, alpha=0.1)
    print(f"Conformal threshold (alpha=0.1): qhat={qhat:.4f}")

    agent = AgenticVerifier(graph, tfidf_vectors, learned_model=model, learned_scaler=scaler, conformal_qhat=qhat)

    print("\nRunning agentic verifier over gold claims...")
    agent_results = evaluate_agent(agent, meta_gold, y_gold)
    print(f"n={agent_results['n']} escalation_rate={agent_results['escalation_rate']:.3f} "
          f"fragile_rate={agent_results['fragile_rate']:.3f} abstention_rate={agent_results['abstention_rate']:.3f}")
    if agent_results["metrics_on_answered"]:
        m = agent_results["metrics_on_answered"]
        print(f"On answered claims (n={agent_results['n_answered']}): "
              f"precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f} "
              f"(F1 CI [{m['f1_ci'][0]:.3f}, {m['f1_ci'][1]:.3f}])")

    rule_preds = np.array([1 if verify(m["paper_a"], m["relation"], m["paper_b"], graph) == "supported" else 0
                            for m in meta_gold])
    rule_metrics = vm.bootstrap_prf1(y_gold, rule_preds)
    print(f"\nFor comparison, plain rule (no agentic loop, answers every claim): "
          f"precision={rule_metrics['precision']:.3f} recall={rule_metrics['recall']:.3f} f1={rule_metrics['f1']:.3f}")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "agentic_verifier_results.json", "w", encoding="utf-8") as f:
        json.dump({"agent": agent_results, "plain_rule_baseline": rule_metrics}, f, indent=2)
    print(f"\nSaved to {out_dir / 'agentic_verifier_results.json'}")


if __name__ == "__main__":
    main()
