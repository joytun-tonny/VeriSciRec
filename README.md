# VeriSciRec & SciRecBench

Code and data for the paper

> **Verifying Relational Claims in Agentic Recommendation Software System via Citation-Graph Grounding**
> Joytun Nessa Tonny, Sajidur Rahman, Saddam Mukta, Md Adnanul Islam
> *Under review at Information and Software Technology (Elsevier).*

## Overview

LLM-based recommenders increasingly justify their rankings with a natural-language
rationale. In scholarly recommendation, these rationales often state how two recommended
papers relate, for example that one *builds on*, *extends* or *supersedes* another.
Existing checks catch a fabricated title or an invented number, but not a **false
relationship between two real papers**. We call this a **relational hallucination**.

This repository contains:

- **VeriSciRec**: a multi-agent recommendation pipeline whose profiling, retrieval,
  generation and verification agents are architecturally separate. The verification
  agent checks each claimed relation `(A, relation, B)` against **citation-graph
  structure** rather than text.
- **SciRecBench**: a benchmark of **10,000 recommendation rationales** across three
  scholarly domains (arXiv CS, ACL Anthology, PubMed), with an expert-annotated gold
  subset (68 instances, 303 claims, mean Cohen's κ = 0.80).

Key results (see the paper for details):

- Text-based verifiers detect **0%** of the confirmed relational hallucinations.
- VeriSciRec recovers the large majority (**F1 = 0.684** on adjudicable claims).
- The gain comes from graph grounding, not from isolating the checker from the generator
  (p = .94 for isolation alone).
- Gating delivery on the verdicts excludes every flagged claim without hurting ranking
  quality against established baselines.

## Repository structure

```
.
├── benchmarks/final/          SciRecBench construction, human annotation, verdicts
│   ├── agentic-nb1*.ipynb       benchmark construction and citation-edge checks
│   ├── agentic-nb2-*.ipynb      expert annotation, agreement, gold merge
│   ├── ANNOTATION_GUIDE.md      annotation guidelines
│   ├── README.md                benchmark construction guide and schema
│   ├── human_eval/              annotation sheets and the three annotators' labels
│   ├── nb2-output/              gold subset, IAA, human-validation results
│   └── relational/              VeriSciRec (CiteCheck) verdicts
├── notebooks/                 verifier, multi-agent system, evaluation (01–10)
│   ├── config.py                canonical paths (local and Kaggle)
│   └── archive/                 superseded notebook
├── novelty-upgrade/           extensions: learned verifier, calibration, hetero-graph, DPO
│   ├── src/                     library code imported by the notebooks
│   └── notebooks/
├── outputs/                   result tables and figures
├── scripts/make_release.sh    builds the compressed data release
├── data/, embeddings/         source corpus and embeddings (not in git, see DATA.md)
├── EXECUTION_ORDER.md         the order to run the notebooks in
└── DATA.md                    what data exists, where to get it, where to put it
```

## Getting started

```bash
git clone https://github.com/joytun-tonny/SciRecBench.git
cd SciRecBench
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Get the data.** The expert-gold subset ships with the repo. Download the full
   benchmark and citation-graph cache from the
   [Releases](https://github.com/joytun-tonny/SciRecBench/releases) page and place them
   as described in [DATA.md](DATA.md).
2. **Set API keys** as environment variables or Kaggle Secrets (`OPENROUTER_API_KEY`,
   `OPENAI_API_KEY`, `HF_TOKEN`, optional `S2_API_KEY`). Only notebooks marked 💲 in
   [EXECUTION_ORDER.md](EXECUTION_ORDER.md) make paid API calls.
3. **Run the notebooks** in the order given in [EXECUTION_ORDER.md](EXECUTION_ORDER.md).
   The paper's experiments were run on Kaggle (Python 3.12, GPU accelerator). Notebooks
   resolve paths automatically on Kaggle (`/kaggle/input`, `/kaggle/working`) and locally
   (`notebooks/config.py`).

To reproduce the core finding only (relational verification vs. text-based verifiers),
run `notebooks/01_citecheck.ipynb` and `notebooks/02-relational-findings.ipynb`. Both are
free to run.

## Benchmark format

Each SciRecBench instance bundles a user profile, a query, a ground-truth paper list,
hard negatives, and an LLM-generated rationale decomposed into claims labelled *atomic*
or *relational* `(A, relation, B)`, with hallucination labels. The full schema and
construction protocol are in [`benchmarks/final/README.md`](benchmarks/final/README.md),
and the annotation protocol is in
[`benchmarks/final/ANNOTATION_GUIDE.md`](benchmarks/final/ANNOTATION_GUIDE.md).

## Citation

The paper is under review. Until it is published, please cite:

```bibtex
@unpublished{tonny2026verisci,
  title  = {Verifying Relational Claims in Agentic Recommendation Software System via Citation-Graph Grounding},
  author = {Tonny, Joytun Nessa and Rahman, Sajidur and Mukta, Saddam and Islam, Md Adnanul},
  note   = {Under review at Information and Software Technology},
  year   = {2026}
}
```

## License

- Code: [MIT](LICENSE)
- SciRecBench data and annotations: [CC BY 4.0](LICENSE-DATA.md)
