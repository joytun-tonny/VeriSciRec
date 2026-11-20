"""
Heterogeneous scholarly evidence graph + lightweight relational GNN (novelty item 5).

SCOPED DOWN from the originally proposed full proposal in three ways, each a real
methodological choice with a real cost/benefit, documented here rather than silently:

1. Entity extraction is HEURISTIC (curated lexicon + regex acronym-pattern matching,
   see entities.py), not LLM-based open extraction. Free, deterministic, reproducible
   byte-for-byte on any re-run (Kaggle or otherwise) with no API key and no sampling
   variance -- at the cost of recall on method/dataset mentions the lexicon doesn't
   cover. entities.py sketches an LLM-based upgrade path (same not-auto-run pattern as
   rationale_revision.py's regeneration step).

2. Node types are {paper, method, dataset} -- "task" nodes are dropped for this first
   pass. Task descriptions in abstracts are free-form phrases ("named entity
   recognition", "long-document summarization"), not proper nouns, so a lexicon/regex
   heuristic would have much lower precision than for methods/datasets, which tend to
   be named with capitalized acronyms or well-known proper nouns. RELATIONS below is
   structured so a task node type and its two edge directions could be added later
   without touching the GNN code.

3. The GNN (RGCNLite) is a minimal hand-rolled 2-layer relational graph convolution in
   plain PyTorch, not torch_geometric/DGL. Reason: this needs to run unmodified on
   whatever Kaggle environment the user has, and torch_geometric's compiled extensions
   are version-locked to specific torch+CUDA builds in a way that silently breaks
   across Kaggle image updates; a ~30-line hand-rolled R-GCN layer (per-relation weight
   matrix, mean aggregation over incoming edges, summed across relations) has no such
   fragility and is easy to audit end to end.

What this buys: whether adding non-citation evidence (shared methods/datasets between
two papers, propagated through the graph by the GNN) improves the verifier over the
citation+text-only 16-feature model from verifier_features.py -- tested the same
silver-train/gold-eval, bootstrap-CI way as items 1 and 2, so directly comparable to
their numbers. Trained via a link-prediction objective (predict silver "supported"
paper-paper pairs against random negative pairs), since there is no separate gold
supervision for "these two papers are graph-related" beyond the relational-claim labels
themselves.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from entities import extract_entities
from graph_utils import CitationGraph, load_relational_claims, verify

RELATIONS = ["cites", "cited_by", "uses_method", "used_by", "evaluates_on", "evaluated_by", "self_loop"]
REL_INDEX = {r: i for i, r in enumerate(RELATIONS)}


class HeteroGraph:
    """In-memory heterogeneous graph: integer node ids shared across all node types,
    parallel `node_type`/`node_name` arrays, and a flat edge list of
    (relation_id, src_id, dst_id) triples."""

    def __init__(self):
        self.node_id: Dict[Tuple[str, str], int] = {}
        self.node_type: List[str] = []
        self.node_name: List[str] = []
        self.edges: List[Tuple[int, int, int]] = []

    def get_or_add_node(self, ntype: str, name: str) -> int:
        key = (ntype, name)
        if key not in self.node_id:
            self.node_id[key] = len(self.node_type)
            self.node_type.append(ntype)
            self.node_name.append(name)
        return self.node_id[key]

    def add_edge(self, rel: str, src_type: str, src_name: str, dst_type: str, dst_name: str):
        s = self.get_or_add_node(src_type, src_name)
        d = self.get_or_add_node(dst_type, dst_name)
        self.edges.append((REL_INDEX[rel], s, d))

    @property
    def n_nodes(self) -> int:
        return len(self.node_type)


def build_hetero_graph(paper_abstracts: Dict[str, str], citation_graph: CitationGraph,
                        max_cite_edges_per_paper: int = 30, seed: int = 0
                        ) -> Tuple[HeteroGraph, Dict[str, Dict[str, Set[str]]]]:
    """
    Builds nodes/edges for every paper with an abstract, their extracted method/dataset
    entities, and citation edges between any two papers that BOTH have abstracts (a
    citation edge to an off-corpus paper ID would create an isolated featureless node,
    not useful for the GNN, so it's excluded rather than added dead-weight).

    max_cite_edges_per_paper caps outgoing citation edges per paper -- a few papers in
    this corpus cite very long reference lists, and unbounded fan-out would dominate
    the message-passing neighborhood for those nodes without proportionally useful
    signal. Kept edges are a fixed-seed sample over a SORTED ref list (not raw set
    iteration, which is not order-stable across Python processes), so the sample is
    reproducible run to run.
    """
    g = HeteroGraph()
    entity_cache: Dict[str, Dict[str, Set[str]]] = {}

    for pid in paper_abstracts:
        g.get_or_add_node("paper", pid)

    for pid, abstract in paper_abstracts.items():
        ents = extract_entities(abstract)
        entity_cache[pid] = ents
        for m in sorted(ents["methods"]):
            g.add_edge("uses_method", "paper", pid, "method", m)
            g.add_edge("used_by", "method", m, "paper", pid)
        for d in sorted(ents["datasets"]):
            g.add_edge("evaluates_on", "paper", pid, "dataset", d)
            g.add_edge("evaluated_by", "dataset", d, "paper", pid)

    rng = random.Random(seed)
    for pid in sorted(paper_abstracts):
        refs = sorted(r for r in citation_graph.refs(pid) if r in paper_abstracts)
        if len(refs) > max_cite_edges_per_paper:
            refs = rng.sample(refs, max_cite_edges_per_paper)
        for ref in refs:
            g.add_edge("cites", "paper", pid, "paper", ref)
            g.add_edge("cited_by", "paper", ref, "paper", pid)

    for i in range(g.n_nodes):
        g.edges.append((REL_INDEX["self_loop"], i, i))

    return g, entity_cache


class RGCNLite(nn.Module):
    """Hand-rolled 2-layer relational graph convolution: one weight matrix per
    relation type, mean-aggregated over each relation's incoming edges, summed across
    relations plus a self-transform, then ReLU. See module docstring for why this is
    hand-rolled instead of torch_geometric/DGL."""

    def __init__(self, n_nodes: int, n_relations: int, in_dim: int = 32,
                 hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, in_dim)
        self.w1 = nn.ModuleList([nn.Linear(in_dim, hidden_dim, bias=False) for _ in range(n_relations)])
        self.w2 = nn.ModuleList([nn.Linear(hidden_dim, out_dim, bias=False) for _ in range(n_relations)])
        self.self1 = nn.Linear(in_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, out_dim)

    @staticmethod
    def _layer(h: torch.Tensor, edges_by_rel: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
               weight_list: nn.ModuleList, self_lin: nn.Linear) -> torch.Tensor:
        n = h.shape[0]
        out = self_lin(h)
        for rel_id, (src, dst) in edges_by_rel.items():
            if src.numel() == 0:
                continue
            msg = weight_list[rel_id](h[src])
            agg = torch.zeros(n, msg.shape[1], device=h.device)
            agg.index_add_(0, dst, msg)
            deg = torch.zeros(n, device=h.device).index_add_(
                0, dst, torch.ones(len(dst), device=h.device)).clamp(min=1).unsqueeze(1)
            out = out + agg / deg
        return F.relu(out)

    def forward(self, edges_by_rel: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        h0 = self.embed.weight
        h1 = self._layer(h0, edges_by_rel, self.w1, self.self1)
        h2 = self._layer(h1, edges_by_rel, self.w2, self.self2)
        return h2


def _edges_by_relation(hg: HeteroGraph, device: str = "cpu") -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    by_rel: Dict[int, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    for rel_id, s, d in hg.edges:
        by_rel[rel_id][0].append(s)
        by_rel[rel_id][1].append(d)
    return {r: (torch.tensor(s, dtype=torch.long, device=device), torch.tensor(d, dtype=torch.long, device=device))
            for r, (s, d) in by_rel.items()}


def positive_pairs_from_claims(claims: List[dict], silver_labels: Dict[Tuple[str, str], int],
                                paper_abstracts: Dict[str, str]) -> List[Tuple[str, str]]:
    """(paper_a, paper_b) pairs from silver-"supported" relational claims, restricted
    to papers that actually have a node in the hetero graph (i.e. have an abstract)."""
    pairs = []
    for c in claims:
        key = (c["instance_id"], c["claim_id"])
        if silver_labels.get(key) != 1:
            continue
        a, b = c["paper_a"], c["paper_b"]
        if a in paper_abstracts and b in paper_abstracts:
            pairs.append((a, b))
    return pairs


def _to_node_id_pairs(hg: HeteroGraph, pairs: List[Tuple[str, str]]) -> List[Tuple[int, int]]:
    out = []
    for a, b in pairs:
        ka, kb = ("paper", a), ("paper", b)
        if ka in hg.node_id and kb in hg.node_id:
            out.append((hg.node_id[ka], hg.node_id[kb]))
    return out


def train_rgcn(hg: HeteroGraph, positive_pairs: List[Tuple[int, int]], n_epochs: int = 30,
                lr: float = 0.01, seed: int = 0, device: str = "cpu") -> Tuple[RGCNLite, np.ndarray]:
    """Link-prediction training: dot-product score between paper embeddings, positive
    pairs from silver-supported claims vs. random-negative paper pairs, binary cross-
    entropy loss. Returns the trained model and the final node embedding matrix."""
    torch.manual_seed(seed)
    edges_by_rel = _edges_by_relation(hg, device=device)

    model = RGCNLite(hg.n_nodes, len(RELATIONS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos = torch.tensor(positive_pairs, dtype=torch.long, device=device)
    n_pos = len(pos)
    paper_node_ids = [i for i, t in enumerate(hg.node_type) if t == "paper"]

    rng = random.Random(seed)
    for epoch in range(n_epochs):
        emb = model(edges_by_rel)
        pos_score = (emb[pos[:, 0]] * emb[pos[:, 1]]).sum(dim=1)
        neg_dst = torch.tensor([rng.choice(paper_node_ids) for _ in range(n_pos)], device=device)
        neg_score = (emb[pos[:, 0]] * emb[neg_dst]).sum(dim=1)
        labels = torch.cat([torch.ones(n_pos), torch.zeros(n_pos)]).to(device)
        scores = torch.cat([pos_score, neg_score])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch}: loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        final_emb = model(edges_by_rel).cpu().numpy()
    return model, final_emb


def hetero_pair_similarity(a: str, b: str, hg: HeteroGraph, embeddings: np.ndarray) -> float:
    """Cosine similarity between the GNN-learned embeddings of papers a and b; 0.0 if
    either paper has no node in the hetero graph (no abstract available)."""
    ka, kb = ("paper", a), ("paper", b)
    if ka not in hg.node_id or kb not in hg.node_id:
        return 0.0
    ea, eb = embeddings[hg.node_id[ka]], embeddings[hg.node_id[kb]]
    denom = float(np.linalg.norm(ea) * np.linalg.norm(eb))
    return float(np.dot(ea, eb) / denom) if denom > 0 else 0.0


def main():
    root = Path(__file__).resolve().parents[2]
    refs_cache = root / "benchmarks/final/relational/citecheck_refs_cache.json"
    cite_results = root / "benchmarks/final/relational/citecheck_results.json"
    bench = root / "benchmarks/final/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"
    bench_verified = root / "benchmarks/final/nb2-output/scireqa_human_verified_subset.jsonl"

    from hard_negatives import build_tfidf_index
    from verifier_features import build_dataset
    import verifier_model as vm
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    print("Loading graph and claims...")
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
    print(f"  {len(paper_abstracts)} papers with abstracts")
    tfidf_vectors = build_tfidf_index(paper_abstracts)

    print("Building heterogeneous graph (entity extraction + citation edges)...")
    hg, entity_cache = build_hetero_graph(paper_abstracts, graph)
    n_methods = sum(1 for t in hg.node_type if t == "method")
    n_datasets = sum(1 for t in hg.node_type if t == "dataset")
    print(f"  nodes: {hg.n_nodes} (papers={len(paper_abstracts)}, methods={n_methods}, datasets={n_datasets})")
    print(f"  edges: {len(hg.edges)}")

    silver = vm.load_silver_labels(cite_results)
    gold = vm.load_gold_labels(bench_verified)
    silver_clean = {k: v for k, v in silver.items() if k not in gold}

    pos_pairs = positive_pairs_from_claims(all_claims, silver_clean, paper_abstracts)
    pos_node_pairs = _to_node_id_pairs(hg, pos_pairs)
    print(f"  positive (silver-supported) paper-pair edges for link prediction: {len(pos_node_pairs)}")

    print("\nTraining R-GCN-lite (link prediction)...")
    model, embeddings = train_rgcn(hg, pos_node_pairs, n_epochs=30, seed=0)

    X_train, y_train, meta_train = build_dataset(all_claims, graph, tfidf_vectors, silver_clean)
    X_gold, y_gold, meta_gold = build_dataset(all_claims, graph, tfidf_vectors, gold)

    def add_hetero_col(X, meta):
        extra = np.array(
            [[hetero_pair_similarity(m["paper_a"], m["paper_b"], hg, embeddings)] for m in meta],
            dtype=np.float32,
        ) if meta else np.zeros((0, 1), dtype=np.float32)
        return np.concatenate([X, extra], axis=1)

    X_train_h = add_hetero_col(X_train, meta_train)
    X_gold_h = add_hetero_col(X_gold, meta_gold)

    results = {}
    rule_preds = np.array([
        1 if verify(m["paper_a"], m["relation"], m["paper_b"], graph) == "supported" else 0
        for m in meta_gold
    ])
    results["rule_based"] = vm.bootstrap_prf1(y_gold, rule_preds)

    scaler = StandardScaler().fit(X_train)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(X_train), y_train)
    lr_preds = lr.predict(scaler.transform(X_gold))
    results["logreg_baseline_16feat"] = vm.bootstrap_prf1(y_gold, lr_preds)

    scaler_h = StandardScaler().fit(X_train_h)
    lr_h = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler_h.transform(X_train_h), y_train)
    lr_h_preds = lr_h.predict(scaler_h.transform(X_gold_h))
    results["logreg_plus_hetero_17feat"] = vm.bootstrap_prf1(y_gold, lr_h_preds)

    for name, r in results.items():
        print(f"{name}: precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1']:.3f} "
              f"(F1 CI [{r['f1_ci'][0]:.3f}, {r['f1_ci'][1]:.3f}])")

    # Paired comparison: same gold claims scored under both, more sensitive than
    # eyeballing the two CIs above for overlap (see paired_bootstrap_diff's docstring)
    # -- this is the direct test of whether the hetero-graph feature actually helps.
    hetero_cmp = vm.paired_bootstrap_diff(y_gold, lr_h_preds, lr_preds)
    results["paired_plus_hetero_vs_baseline"] = hetero_cmp
    sig = " *SIGNIFICANT*" if hetero_cmp["f1_significant"] else ""
    print(f"\nPaired +hetero-vs-baseline: F1 diff={hetero_cmp['f1_diff']:+.3f} "
          f"CI=[{hetero_cmp['f1_diff_ci'][0]:+.3f}, {hetero_cmp['f1_diff_ci'][1]:+.3f}]{sig}")

    results["graph_stats"] = {
        "n_nodes": hg.n_nodes, "n_papers": len(paper_abstracts), "n_methods": n_methods,
        "n_datasets": n_datasets, "n_edges": len(hg.edges), "n_positive_pairs": len(pos_node_pairs),
    }

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "hetero_graph_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    torch.save(model.state_dict(), out_dir / "rgcn_lite.pt")
    print(f"\nSaved results to {out_dir / 'hetero_graph_results.json'}")
    print(f"Saved model weights to {out_dir / 'rgcn_lite.pt'}")


if __name__ == "__main__":
    main()
