# Execution order

Run the notebooks in this order. 💲 = makes paid LLM API calls (OpenRouter / OpenAI);
"free" = no API cost. All notebooks were run on Kaggle (Python 3.12, GPU where noted).

API keys are read from Kaggle Secrets or environment variables (`OPENROUTER_API_KEY`,
`OPENAI_API_KEY`, `HF_TOKEN`, optional `S2_API_KEY` / `SEMANTIC_SCHOLAR_API_KEY` for
Semantic Scholar, optional `TAVILY_API_KEY`). Nothing is hardcoded.

## Phase 0: Benchmark construction (SciRecBench)
- `benchmarks/final/agentic-nb1-data-exploration-audit.ipynb`: builds the benchmark with **relational** rationales (Llama-3.3-70B). 💲
- `benchmarks/final/agentic-nb1c-citation-edge-check.ipynb`: citation-edge feasibility gate (arXiv + ACL/PubMed). free
- `benchmarks/final/agentic-nb1b-label-diagnosis-fix.ipynb`: *atomic* hallucination diagnosis and negatives (atomic baseline). free

## Phase 1: Relational verification (core finding)
1. `notebooks/01_citecheck.ipynb`: **VeriSciRec / CiteCheck**, graph-grounded verification of `(A, relation, B)` claims → `citecheck_results.json`. Embedding-free; uses the Semantic Scholar graph. free
2. `benchmarks/final/agentic-nb2-human-annotation.ipynb`: expert relational gold (3 annotators, 68 instances / 303 claims) and agreement. (Optional LLM-judge 💲)
3. `notebooks/02-relational-findings.ipynb`: text-based verifiers vs. VeriSciRec, atomic vs. relational, scored against the expert gold. free

## Phase 2: Multi-agent recommendation system
4. `notebooks/03_embeddings_vector_store.ipynb`: e5-large embeddings + ChromaDB/BM25. GPU, ~30–90 min
5. `notebooks/04_user_profile_agent.ipynb`: user-profile agent
6. `notebooks/05_hybrid_retrieval_agent.ipynb`: hybrid (dense + BM25) retrieval (needs #4)
7. `notebooks/06_critic_generator_agent.ipynb`: generation agent and checker. 💲
8. `notebooks/07_maper_orchestration.ipynb`: multi-agent orchestration (LangGraph) and ablation. 💲

## Phase 3: Mitigation and evaluation
9. `notebooks/08_dpo_pipeline.ipynb`: graph-grounded DPO mitigation. GPU-heavy, 💲
10. `notebooks/09_evaluation_ablation.ipynb`: final evaluation and ablations
11. `notebooks/10-recommendation-quality-eval.ipynb`: ranking quality against established baselines

## Extensions
- `novelty-upgrade/`: learned verifier, relation experts, conformal calibration, heterogeneous graph, agentic verifier, and DPO. See [`novelty-upgrade/README.md`](novelty-upgrade/README.md).

## Archived
- `notebooks/archive/04_citation_graph_gnn_ORIGINAL.ipynb`: the original combined citation-graph + GNN recommender + CiteCheck notebook, superseded by `01_citecheck`.

## Kaggle notes
- Each notebook writes to `/kaggle/working`. To feed the next one, save its outputs as a Kaggle dataset and attach it. Notebooks look for their inputs under `/kaggle/input` automatically where possible.
- Locally, paths resolve through `notebooks/config.py` (`data/`, `embeddings/` and `outputs/` at the repository root).
