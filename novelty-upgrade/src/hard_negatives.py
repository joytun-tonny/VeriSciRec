"""
Hard-negative relational claim construction (novelty item 4).

Random negative pairs are too easy: two topically unrelated papers asserted to have
some relationship are trivially rejectable by almost anything. These strategies
instead corrupt a REAL, plausible claim in a targeted way, so the resulting negative
still "looks like" something a fluent LLM rationale would write.

Strategies implemented here, and why each one is scoped the way it is:

  direction_reversal        -- swap (A,rho,B) -> (B,rho,A) for a real dependency
                                claim. Directly motivated by this session's
                                counterfactual finding: the current rule-based
                                verifier is nearly blind to this (19.1% flip rate on
                                dependency relations, 0% on compares/shares_topic),
                                so these are the single most diagnostic negatives to
                                have on hand.
  relation_substitution     -- swap rho to a different taxonomy member for a real
                                claim's (A, B) pair.
  shared_refs_no_dependency -- real pairs with high co-citation but NO direct
                                citation edge, asserted as a FALSE dependency
                                relation. Structurally precise: these pairs pass
                                compares/shares_topic (e_AB=1 via co-citation) but
                                MUST fail builds_on/uses_method_of/supersedes (which
                                require a direct edge), so this negative type is
                                verifiable as false by construction, not by guesswork.
  temporally_impossible     -- a dependency claim (A, builds_on, B) where A
                                chronologically predates B, so A could not have
                                built on B.
  topically_similar_swap    -- replace B with the paper most similar to it BY TEXT
                                (TF-IDF over the abstracts already in the benchmark,
                                not the arXiv-only dense embedding index, so this
                                works uniformly across all three domains) among the
                                OTHER candidates in the same instance, excluding B
                                itself.
  same_author_swap          -- (arXiv only -- the only domain with a local author
                                field) replace B with a different paper by an
                                overlapping author, from data/papers_metadata_10k_backup.csv.

Deferred, not built here, with reasons:
  same_venue        -- no venue field is populated for any domain in current data;
                        would need a Semantic Scholar venue fetch pass first.
  citation_intent_mismatch -- needs the citation-intent classifier (Section 3.2's
                        optional refinement), which is not yet built/enabled.

Every generated negative is scored against the CURRENT rule-based verify(), so this
module doubles as a diagnostic: for each strategy, what fraction does the existing
verifier already catch (mark unsupported) vs. get fooled by (mark supported)? That
fooled-rate is exactly what a learned verifier (item 1) needs to close.

IMPORTANT ASYMMETRY, read before using "fooled_rate" as a training/reporting number:
direction_reversal, relation_substitution, shared_refs_no_dependency, and
temporally_impossible are guaranteed false BY CONSTRUCTION (wrong direction, wrong
relation type, no citation edge exists, or chronologically impossible) -- for these,
"fooled" cleanly means "the verifier is wrong." topically_similar_swap and
same_author_swap are NOT guaranteed false: a topically similar or same-author paper
can genuinely be independently connected too (small-world citation structure), so
part of their measured "fooled rate" (45.3% / 77.2% in the first real run) reflects
the swap accidentally landing on a real relationship, not a verifier weakness. Before
using these two strategies' outputs as training negatives, filter further (e.g., also
require shared_refs(a, swapped_b) below some threshold) rather than trusting
caught_by_rule_based_verifier alone to mean "definitely a false claim."
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from graph_utils import CitationGraph, DEPENDENCY_RELATIONS, NONDEPENDENCY_RELATIONS, verify, TAU_DEFAULT

ALL_RELATIONS = DEPENDENCY_RELATIONS | NONDEPENDENCY_RELATIONS


@dataclass
class HardNegative:
    strategy: str
    source_instance_id: str
    source_claim_id: str
    paper_a: str
    relation: str
    paper_b: str
    domain: str
    caught_by_rule_based_verifier: bool  # True = verify() correctly says unsupported
    rule_based_verdict: str


# -- Strategy 1: direction reversal --------------------------------------------------

def gen_direction_reversal(claims: List[dict], graph: CitationGraph, rng: random.Random) -> List[HardNegative]:
    out = []
    for c in claims:
        a, rho, b = c["paper_a"], c["relation"], c["paper_b"]
        if rho not in DEPENDENCY_RELATIONS or not a or not b:
            continue
        if c["recorded_verdict"] != "supported":
            continue  # only reverse claims that were genuinely true in the right direction
        v = verify(b, rho, a, graph)  # swapped
        out.append(HardNegative("direction_reversal", c["instance_id"], c["claim_id"],
                                 b, rho, a, c["domain"], v == "unsupported", v))
    return out


# -- Strategy 2: relation substitution -----------------------------------------------

def gen_relation_substitution(claims: List[dict], graph: CitationGraph, rng: random.Random) -> List[HardNegative]:
    out = []
    for c in claims:
        a, rho, b = c["paper_a"], c["relation"], c["paper_b"]
        if not (a and rho and b) or c["recorded_verdict"] != "supported":
            continue
        alt = rng.choice(sorted(ALL_RELATIONS - {rho}))
        v = verify(a, alt, b, graph)
        out.append(HardNegative("relation_substitution", c["instance_id"], c["claim_id"],
                                 a, alt, b, c["domain"], v == "unsupported", v))
    return out


# -- Strategy 3: shared references, no dependency ------------------------------------

def gen_shared_refs_no_dependency(claims: List[dict], graph: CitationGraph, rng: random.Random,
                                   tau: int = TAU_DEFAULT, max_per_domain: int = 2000) -> List[HardNegative]:
    """
    Scan claim pairs' papers for (A, B) with shared_refs >= tau but no direct edge in
    either direction -- these are real co-citation pairs, verifiably NOT a dependency.
    """
    out = []
    seen = set()
    by_domain_count = defaultdict(int)
    pool = list(claims)
    rng.shuffle(pool)
    for c in pool:
        a, b, dom = c["paper_a"], c["paper_b"], c["domain"]
        if not a or not b or a == b or (a, b) in seen or by_domain_count[dom] >= max_per_domain:
            continue
        seen.add((a, b))
        if a not in graph or b not in graph:
            continue
        if graph.cites(a, b) or graph.cites(b, a):
            continue  # a direct edge exists -- not this strategy's target case
        if graph.shared_refs(a, b) < tau:
            continue
        rho = rng.choice(sorted(DEPENDENCY_RELATIONS))  # assert a FALSE dependency
        v = verify(a, rho, b, graph, tau=tau)
        out.append(HardNegative("shared_refs_no_dependency", c["instance_id"], c["claim_id"],
                                 a, rho, b, dom, v == "unsupported", v))
        by_domain_count[dom] += 1
    return out


# -- Strategy 4: temporally impossible -----------------------------------------------

def gen_temporally_impossible(claims: List[dict], graph: CitationGraph, rng: random.Random) -> List[HardNegative]:
    out = []
    for c in claims:
        a, rho, b = c["paper_a"], c["relation"], c["paper_b"]
        if rho not in DEPENDENCY_RELATIONS or a not in graph or b not in graph:
            continue
        ya, yb = graph.year(a), graph.year(b)
        if ya is None or yb is None or ya >= yb:
            continue  # not actually temporally impossible for A to depend on B
        # A strictly predates B -- claiming (A, dependency, B) is chronologically impossible
        v = verify(a, rho, b, graph)
        out.append(HardNegative("temporally_impossible", c["instance_id"], c["claim_id"],
                                 a, rho, b, c["domain"], v == "unsupported", v))
    return out


# -- Strategy 5: topically-similar swap (TF-IDF over benchmark abstracts) -----------

_WORD_RE = re.compile(r"[a-z]{3,}")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall((text or "").lower())


def build_tfidf_index(paper_abstracts: Dict[str, str]):
    """Minimal, dependency-free TF-IDF (no sklearn requirement) over paper abstracts."""
    import math
    docs = {pid: _tokenize(txt) for pid, txt in paper_abstracts.items()}
    df = defaultdict(int)
    for toks in docs.values():
        for t in set(toks):
            df[t] += 1
    n_docs = max(1, len(docs))
    idf = {t: math.log(n_docs / (1 + c)) + 1 for t, c in df.items()}

    vectors = {}
    for pid, toks in docs.items():
        tf = defaultdict(int)
        for t in toks:
            tf[t] += 1
        vec = {t: (cnt / len(toks)) * idf.get(t, 0.0) for t, cnt in tf.items()} if toks else {}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors[pid] = {t: v / norm for t, v in vec.items()}
    return vectors


def _cosine(v1: dict, v2: dict) -> float:
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(w * v2.get(t, 0.0) for t, w in v1.items())


def gen_topically_similar_swap(claims: List[dict], graph: CitationGraph, rng: random.Random,
                                paper_abstracts: Dict[str, str], tfidf_vectors: dict,
                                candidates_by_domain: Dict[str, List[str]],
                                max_candidates: int = 60) -> List[HardNegative]:
    out = []
    for c in claims:
        a, rho, b, dom = c["paper_a"], c["relation"], c["paper_b"], c["domain"]
        if not (a and rho and b) or b not in tfidf_vectors:
            continue
        pool = candidates_by_domain.get(dom, [])
        if len(pool) > max_candidates:
            pool = rng.sample(pool, max_candidates)
        best_pid, best_sim = None, -1.0
        for cand in pool:
            if cand in (a, b) or cand not in tfidf_vectors:
                continue
            sim = _cosine(tfidf_vectors[b], tfidf_vectors[cand])
            if sim > best_sim:
                best_pid, best_sim = cand, sim
        if best_pid is None:
            continue
        v = verify(a, rho, best_pid, graph)
        out.append(HardNegative("topically_similar_swap", c["instance_id"], c["claim_id"],
                                 a, rho, best_pid, dom, v == "unsupported", v))
    return out


# -- Strategy 6: same-author swap (arXiv only) ---------------------------------------

def load_arxiv_authors(csv_path: Path, limit: Optional[int] = None) -> Dict[str, set]:
    authors_by_paper = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            raw = row.get("authors", "")
            names = {n.strip().lower() for n in re.split(r",| and ", raw) if n.strip()}
            if names:
                authors_by_paper[row["arxiv_id"]] = names
    return authors_by_paper


def gen_same_author_swap(claims: List[dict], graph: CitationGraph, rng: random.Random,
                          authors_by_paper: Dict[str, set]) -> List[HardNegative]:
    by_author = defaultdict(set)
    for pid, names in authors_by_paper.items():
        for n in names:
            by_author[n].add(pid)

    out = []
    for c in claims:
        a, rho, b, dom = c["paper_a"], c["relation"], c["paper_b"], c["domain"]
        if dom != "arxiv_cs" or not (a and rho and b) or b not in authors_by_paper:
            continue
        candidates = set()
        for n in authors_by_paper[b]:
            candidates |= by_author[n]
        candidates -= {a, b}
        if not candidates:
            continue
        swap = rng.choice(sorted(candidates))
        v = verify(a, rho, swap, graph)
        out.append(HardNegative("same_author_swap", c["instance_id"], c["claim_id"],
                                 a, rho, swap, dom, v == "unsupported", v))
    return out


def summarize(negatives: List[HardNegative]) -> dict:
    by_strategy = defaultdict(lambda: [0, 0])
    for n in negatives:
        s = by_strategy[n.strategy]
        s[0] += 1
        s[1] += int(n.caught_by_rule_based_verifier)
    return {
        strat: {"n": n, "caught_by_current_verifier": caught, "fooled_rate": 1 - caught / n if n else None}
        for strat, (n, caught) in by_strategy.items()
    }


if __name__ == "__main__":
    from graph_utils import load_relational_claims

    root = Path(__file__).resolve().parents[2]
    refs_cache = root / "benchmarks/final/relational/citecheck_refs_cache.json"
    cite_results = root / "benchmarks/final/relational/citecheck_results.json"
    bench = root / "benchmarks/final/nb2-output/scireqa_full_relational_v1.1_humanmerged.jsonl"
    arxiv_meta = root / "data/papers_metadata_10k_backup.csv"

    graph = CitationGraph.load(refs_cache)
    claims = load_relational_claims(bench, cite_results)
    claims = [c for c in claims if c["recorded_verdict"] is not None]
    print(f"Loaded {len(claims)} relational claims, graph has {len(graph)} nodes.")

    rng = random.Random(0)
    all_negatives: List[HardNegative] = []

    all_negatives += gen_direction_reversal(claims, graph, rng)
    all_negatives += gen_relation_substitution(claims, graph, rng)
    all_negatives += gen_shared_refs_no_dependency(claims, graph, rng)
    all_negatives += gen_temporally_impossible(claims, graph, rng)

    # Topically similar swap needs abstracts -- pull them from the benchmark's own
    # ground_truth_papers entries (present for all three domains) rather than the
    # arXiv-only dense embedding index.
    print("Building TF-IDF index over benchmark abstracts...")
    paper_abstracts: Dict[str, str] = {}
    candidates_by_domain: Dict[str, set] = defaultdict(set)
    with open(bench, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            dom = rec.get("domain")
            for gt in rec.get("ground_truth_papers") or []:
                pid, ab = gt.get("paper_id"), gt.get("abstract")
                if pid and ab:
                    paper_abstracts[pid] = ab
                    candidates_by_domain[dom].add(pid)
    tfidf_vectors = build_tfidf_index(paper_abstracts)
    candidates_by_domain = {k: list(v) for k, v in candidates_by_domain.items()}
    print(f"  {len(paper_abstracts)} papers with abstracts, domains: "
          f"{ {k: len(v) for k, v in candidates_by_domain.items()} }")
    all_negatives += gen_topically_similar_swap(claims, graph, rng, paper_abstracts,
                                                 tfidf_vectors, candidates_by_domain)

    print("Loading arXiv author metadata...")
    authors_by_paper = load_arxiv_authors(arxiv_meta)
    print(f"  {len(authors_by_paper)} arXiv papers with author lists")
    all_negatives += gen_same_author_swap(claims, graph, rng, authors_by_paper)

    print(f"\nGenerated {len(all_negatives)} hard negatives across "
          f"{len(set(n.strategy for n in all_negatives))} strategies.\n")

    summary = summarize(all_negatives)
    print(f"{'strategy':26s} {'n':>7s} {'caught':>8s} {'fooled rate':>12s}")
    for strat, s in summary.items():
        print(f"{strat:26s} {s['n']:7d} {s['caught_by_current_verifier']:8d} {s['fooled_rate']:12.1%}")

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "hard_negatives.jsonl", "w", encoding="utf-8") as f:
        for n in all_negatives:
            f.write(json.dumps(asdict(n)) + "\n")
    with open(out_dir / "hard_negatives_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {len(all_negatives)} negatives to {out_dir / 'hard_negatives.jsonl'}")
    print(f"Saved summary to {out_dir / 'hard_negatives_summary.json'}")
