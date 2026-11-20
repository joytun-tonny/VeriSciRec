# SciRecQA v1.1 — Benchmark Construction Guide

A multi-source benchmark for **trustworthy, verifiable scientific paper recommendation**,
built for the MAPER project. This document explains, end to end,
how the benchmark is constructed, every design decision and *why* it was made, how to run
the pipeline, and how the quality is guaranteed. It is written to be read by a supervisor
or reviewer with no prior context.

---

## 1. What SciRecQA is

Each **instance** poses a realistic recommendation problem:

> *Given a user (their reading history + a query), which papers should be recommended,
> and is the natural-language rationale for those recommendations factually grounded?*

So every instance bundles four things:
1. a **user profile** (papers the user has read),
2. a **query** (e.g. "I recently read X, recommend similar papers"),
3. a **ground-truth (GT) paper list** — the papers that *should* be recommended,
4. a **reference rationale** explaining the recommendation, with **sentence-level
   hallucination labels** (is each claim grounded in the evidence?).

This makes SciRecQA usable for three evaluations at once: **recommendation quality**
(NDCG/Recall/MRR over GT + hard negatives), **rationale faithfulness / hallucination
detection**, and **personalization**.

**Scale:** 10,000 instances across three domains for cross-domain generalizability:

| Domain | Instances | Source | Role |
|---|---|---|---|
| arXiv CS | 6,000 | real user ratings (`rated_papers.csv`) | primary, real interactions |
| ACL Anthology | 2,000 | Semantic Scholar API | NLP in-domain |
| PubMed | 2,000 | Semantic Scholar API | biomedical, **held-out zero-shot** |

---

## 2. Why v1.1 exists (the audit that motivated a rebuild)

The first version (v1.0, generated with GPT-4o-mini) was audited against the raw data and
found to have **correctness defects that would invalidate the evaluation**. v1.1 fixes them.
This table is the heart of the contribution — it is what makes the benchmark defensible:

| Problem found in v1.0 | Measured (v1.0) | Fix in v1.1 | Result (verified) |
|---|---|---|---|
| **GT = the user's own visible profile** (data leakage) | 100% of arXiv GT papers were already shown in the input profile | GT is a *temporal hold-out* disjoint from the profile | **0% overlap** |
| **Disliked papers used as GT** | 22,683 GT papers (39.5%) were rated **−1** by the user | filter to `rating > 0` before building GT | **0 negative-rated GT** |
| **Empty queries** | 564 arXiv queries (9.4%) were blank | fallback cascade (title→abstract→categories→id) | **0 empty queries** |
| **Low query diversity** | 0.809 (below the 0.9 bar) | personalization + deterministic tiebreaker | **0.991** |
| **Uninformative labels** | ~95% of claims labelled generic "neutral" | claim-type gating + a real fact-checker | informative claims rise sharply |
| **Fixed GT size** | every instance had exactly 10 GT papers | sample GT size in [3, 8] | **variance restored** |
| **No hard negatives / difficulty** | absent | embedding-KNN hard negatives + difficulty score | **5 hard negatives/instance** |

(Numbers above were verified by re-running the v1.1 construction on the existing local data.
Final figures come from the Kaggle run.)

---

## 3. Instance schema (v1.1)

```jsonc
{
  "id": "<12-char md5>", "benchmark_version": "1.1", "construction_date": "YYYY-MM-DD",
  "domain": "arxiv_cs | acl_anthology | pubmed",
  "split":  "train | val | test",
  "annotation_status": "auto_labeled | pending_human | human_verified",

  "user_id": "...",
  "user_profile": { "papers": [...], "titles": [...], "abstracts": [...] },

  "query": "<=200 chars", "query_type": "history_based | natural_language | hybrid",
  "original_query": "...", "query_paper_id": "...",

  "ground_truth_papers": [ {paper_id, title, abstract, year, venue, authors} ],

  // ---- v1.1 additions ----
  "hard_negatives":   [ {paper_id, title, score, source} ],   // 5 topically-similar non-GT papers
  "difficulty_score": 0.18,                                   // higher = harder; null if no embedding
  "query_quality":    {length, specificity, type_fit, overall},

  "reference_rationale": "<Llama-3.3-70B, grounded in abstracts>",
  "rationale_metadata":  {model, served_by, generated, generation_date},

  "hallucination_labels": [
    { "claim_id", "claim_text",
      "claim_type": "factual_candidate | general_statement",
      "verdict": "supported | unsupported | neutral | general_statement | pending_annotation",
      "confidence", "label_type", "model", "human_verified", "human_label" }
  ],
  "entity_verification": { total_cited, verified, unverified, citation_trust_score }
}
```

---

## 4. Construction pipeline (what each step does and *why*)

The pipeline lives in **`agentic-nb1-data-exploration-audit.ipynb`** (construction) and
**`agentic-nb2-human-annotation.ipynb`** (human validation).

### 4.1 Load + audit (nb1 §3–7)
Loads `papers_metadata.csv`, `rated_papers.csv`, `implicit_interactions.csv`, the precomputed
1024-d paper embeddings, and JSON metadata. Produces a data-quality report.

### 4.2 arXiv instance construction — the core fix (nb1 §14)
For each user:
1. **Keep only positively-rated papers** (`rating > 0`). A paper the user disliked must
   never be a "correct" recommendation.
2. Sort their positives chronologically.
3. **Temporal hold-out split:** the *earlier* positives become the **visible profile**;
   a *later, disjoint* band (size sampled in [3, 8]) becomes the **ground truth**; the most
   recent positive becomes the **query paper**.
   - *Why:* the GT is no longer visible in the prompt (no leakage), and every GT paper
     precedes the query in time (no future leakage). An invariant assert enforces
     `GT ∩ profile = ∅` for every instance.
4. Users with too few positives are dropped (correct, not a regression).

### 4.3 ACL + PubMed construction (nb1 §15–16)
Papers are fetched from Semantic Scholar. Profiles are simulated from adjacent/clustered
papers; GT is drawn from **future** papers (ACL) or same-venue papers (PubMed), with the same
guards: GT size sampled in [3, 8], GT excludes profile papers, and `GT year ≤ query year`.

### 4.4 Query generation (nb1 §9)
Produces ecologically valid queries with a **fallback cascade** so a query is never empty:
`title + first abstract sentence → categories/venue → arXiv id → profile keyword`.

### 4.5 Enrichment pass (nb1 §16.5)
Per instance, adds:
- **`hard_negatives`** — 5 papers most similar to the query *by embedding* but **not** in
  GT/profile. *Why embeddings and not categories:* 96% of the metadata has no category field,
  and embedding-nearest-neighbours are stronger "hard" negatives anyway (cosine ~0.88–0.91).
- **`difficulty_score`** — inverse similarity between the query and the GT centroid
  (embedding-based for arXiv; lexical fallback for ACL/PubMed). Enables difficulty-stratified
  reporting.
- **`query_quality`** — rule-based sub-scores (length / specificity / type-fit).
- A **deterministic tiebreaker** guarantees unique queries (diversity ≥ 0.95).

### 4.6 Reference rationale generation (nb1 §13, §17)
A grounded 3–4 sentence rationale per instance, generated by **Llama-3.3-70B-Instruct via
OpenRouter** (see §6 for the model rationale). Checkpointed and resumable.

### 4.7 Hallucination labeling (nb1 §11, §18)
1. Split the rationale into atomic claims.
2. **Claim-type gating:** classify each claim as `factual_candidate` (has a quote, year,
   metric, number, or proper noun) or `general_statement` (generic prose). Only factual
   claims are sent to a model; generic prose is labelled `general_statement` by rule.
   *Why:* this is what fixed the v1.0 "95% neutral" problem — generic sentences were being
   fed to NLI pointlessly.
3. Score factual claims with **MiniCheck** (a claim-grounded-in-document fact-checker) and
   map its support probability to `supported / unsupported / neutral` using thresholds that
   are **calibrated on the human gold subset** (see §5). A strong NLI model is the fallback.

### 4.8 Save + validate (nb1 §20–22)
Writes the JSONL files and runs validators that **must all PASS**: 0 empty queries,
diversity ≥ 0.95, `GT ∩ profile = ∅`, 0 negative-rated GT, GT-size variance present, and
ID uniqueness.

---

## 5. Human validation — the quality anchor (nb2)

Auto labels are **silver**; the human-verified subset is **gold**. This is the section
reviewers scrutinise most.

1. **Stratified sample** of 500 instances (domain × difficulty tercile).
2. **CSV export** for 3 independent annotators with a clear rubric (the `auto_verdict` is
   hidden to avoid anchoring bias).
3. **Inter-annotator agreement:** pairwise **Cohen's κ** + **Krippendorff's α** (target κ ≳ 0.65).
4. **Majority-vote write-back:** sets `human_verified=True`, `human_label`, and
   `annotation_status="human_verified"` on the subset.
5. **Threshold calibration:** sweeps the MiniCheck thresholds against the human gold and
   reports the optimal values to copy back into nb1.
6. **Belt-and-suspenders LLM judge (optional):** an *independent* model (Qwen2.5-72B —
   deliberately **not** the Llama generator, to avoid self-preference bias) re-labels the
   verified subset, and a **3-way agreement** (MiniCheck ↔ judge ↔ human) is reported.

This yields the paper's **Table 5** (human validation + agreement metrics).

---

## 6. Key methodological choices (defending each to a reviewer)

| Choice | What | Why it's defensible |
|---|---|---|
| **Generator** | Llama-3.3-70B-Instruct (open weights) via OpenRouter | Open, reproducible, strong scientific reasoning; better than a small distilled model. |
| **Single host pin** | OpenRouter routed to one provider (DeepInfra), `allow_fallbacks=False`, fixed quantization, `seed` | All 10k rationales share identical serving/quantization — no silent heterogeneity. The host actually used is logged in `served_by`. |
| **Temperature 0.4** | not greedy | A *gold* benchmark of grounded rationales still needs a realistic spread of grounded/ungrounded claims; τ=0 over-grounds and starves the hallucination class. Dataset reproducibility = releasing the fixed JSONL + config, not re-generating. |
| **Labeler = MiniCheck** | claim-grounded-in-document fact-checker (Tang et al. 2024) | Matches the task (claim vs document) better than generic MNLI; efficient; reproducible open weights; calibrated to human gold. |
| **Auto = silver, human = gold** | NLI/MiniCheck labels are weak supervision | The benchmark's authority comes from human agreement, not the model brand. |
| **Hard negatives via embeddings** | 5 nearest non-GT papers | Enables ranking evaluation; categories are missing in 96% of metadata. |

---

## 7. How to run (Kaggle)

> Run on **Kaggle** (GPU + the dataset paths). The labeling step needs `transformers`/`torch`.
> MiniCheck auto-installs in the dependency cell **from GitHub** (`Liyan06/MiniCheck` — it is
> not on PyPI), so **notebook Internet must be ON**. Cell 1 prints `MiniCheck available: True/False`;
> if `False`, labeling falls back to the strong NLI model instead.

1. **Secrets** (Add-ons → Secrets): `OPENROUTER_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY`.
   You need OpenRouter **credits** (the pinned provider is paid, not a `:free` endpoint).
2. **Confirm** the pinned provider (`OPENROUTER_PROVIDER` in cell 3) actually serves
   `meta-llama/llama-3.3-70b-instruct`; otherwise switch to `Together`/`Fireworks`.
3. Run **in stages**:
   - **Stage A (smoke test):** set `TARGET_ARXIV/ACL/PUBMED = 20`, run all. Confirm
     `Serving hosts used: {one host}` and the validators PASS. Cheap sanity check.
   - **Stage B (full construction):** reset targets to 6000/2000/2000, run. All validators PASS.
   - **Stage C (rationales):** 10k Llama-70B calls. These run **concurrently**
     (`GEN_CONCURRENCY=8` in cell 3), which cuts wall-clock from ~10–16 h sequential to
     **~1–2 h** — keeping the whole notebook inside Kaggle's 12 h session limit. Concurrency
     changes only timing, not content/cost (same model/prompt/temperature/seed/provider pin
     per request). It is checkpointed; interrupting and re-running cell §17 resumes. To
     survive across *separate* sessions, save `rationales_checkpoint_v1_1.json` as a Kaggle
     Dataset output and re-attach it as input.
4. Run **`agentic-nb2`** for the 500-instance human study, then copy the calibrated MiniCheck
   thresholds back into nb1 and re-run labeling (cell §18) if they changed.

**Before a clean run:** *Kernel → Restart & Clear Output → Run All.* If you changed any
generation setting, **delete `rationales_checkpoint_v1_1.json`** first, or stale cached
rationales will be reused.

---

## 8. Outputs

| File | Contents |
|---|---|
| `scireqa_arxiv_v1.1.jsonl` / `_acl_` / `_pubmed_` | per-domain instances |
| `scireqa_full_v1.1.jsonl` | all 10k combined |
| `rationales_checkpoint_v1_1.json` | resumable rationale cache (keyed by instance id) |
| `benchmark_stats.json` | per-split statistics (counts, GT size, difficulty, % informative) |
| `annotation/iaa_summary.json` | Cohen κ + Krippendorff α |
| `annotation/minicheck_calibration.json` | calibrated thresholds |
| `annotation/threeway_agreement.json` | MiniCheck ↔ judge ↔ human κ (if judge run) |
| `annotation/scireqa_human_verified_subset.jsonl` | the gold subset |
| `annotation/table5_human_validation.csv` | paper Table 5 |

---

## 9. Reproducibility checklist (for the paper)

- Generator: `meta-llama/llama-3.3-70b-instruct`, `temperature=0.4`, `seed=42`, single pinned
  OpenRouter host (logged per instance in `rationale_metadata.served_by`).
- Labeler: `MiniCheck-flan-t5-large`; thresholds reported (calibrated on human gold).
- Random seed `42` fixed for all sampling (profile/GT split, hard negatives, annotation sample).
- All construction code in the two notebooks; data sources and versions listed above.

---

## 10. Honest limitations (state these in the paper)

- **ACL/PubMed users are simulated** (not real interaction logs); only arXiv uses real ratings.
- **Auto (silver) labels are noisy**; only the 500-instance human subset is gold. Bulk metrics
  should be read as silver unless on the verified subset.
- **Organic hallucination rate is low** at τ=0.4 with a strong model — report the natural rate
  rather than forcing class balance; do not inject fabrications into the gold rationales.
- **Difficulty for ACL/PubMed is a lexical proxy** (no embeddings for Semantic Scholar ids).

---

## 11. References to cite

**Inter-annotator agreement (required):** Cohen (1960); Krippendorff (2011); Fleiss (1971);
Artstein & Poesio (2008).
**Comparator hallucination/factuality benchmarks:** RAGTruth (Niu et al. 2024); FActScore
(Min et al. 2023); HaluEval (Li et al. 2023); TruthfulQA (Lin et al. 2022); FELM (Chen et al.
2023); FEVER (Thorne et al. 2018); SelfCheckGPT (Manakul et al. 2023).
**Dataset documentation norms:** Datasheets (Gebru et al. 2021); Data Statements (Bender &
Friedman 2018); Bowman & Dahl (2021).
**Components used:** MiniCheck (Tang et al. 2024); DeBERTa-v3 / MNLI (He et al.; Williams et
al. 2018); Llama 3 (Dubey et al. 2024); SPECTER (Cohan et al. 2020); DPR (Karpukhin et al.
2020); S2ORC (Lo et al. 2020).

*(Verify exact venues/years before adding to the .bib.)*

---

## 12. File map

```
benchmarks/
└── final/
    ├── agentic-nb1-data-exploration-audit.ipynb   # construction + auto-labeling
    ├── agentic-nb1b-label-diagnosis-fix.ipynb      # atomic hallucination detection fix (AUC 0.949)
    ├── agentic-nb1c-citation-edge-check.ipynb      # citation-edge diagnostic
    ├── agentic-nb2-human-annotation.ipynb          # human validation, IAA, calibration, judge
    ├── README.md                                    # this file
    ├── ANNOTATION_GUIDE.md                          # annotator rubric
    ├── after_diagnosis/                             # FINAL v1.1 outputs (post label-diagnosis fix)
    │   ├── scireqa_full_v1.1.FINAL.jsonl
    │   ├── scireqa_full_with_negatives_v1.1.FINAL.jsonl
    │   └── scireqa_hallucination_eval_v1.1.FINAL.jsonl
    ├── relational/                                  # CiteCheck relational benchmark (headline contribution)
    │   ├── citecheck_refs_cache.json
    │   ├── citecheck_results.json
    │   ├── rationales_checkpoint_relational_v1_1.json
    │   ├── scireqa_full_relational_v1.1.jsonl
    │   ├── scireqa_acl_relational_v1.1.jsonl
    │   ├── scireqa_arxiv_relational_v1.1.jsonl
    │   └── scireqa_pubmed_relational_v1.1.jsonl
    ├── human_eval/                                  # raw human annotation data
    │   ├── claims_master.csv, sample_ids.json
    │   ├── to_annotate_annotator_{A,B,C}.csv
    │   └── done_annotation/done_annotator_{A,B,C}.csv
    └── nb2-output/                                  # human validation results (§5, paper Table 5)
        ├── iaa_summary.json, minicheck_calibration.json
        ├── scireqa_full_relational_v1.1_humanmerged.jsonl
        ├── scireqa_human_verified_subset.jsonl
        └── table5_human_validation.csv
```

Superseded material (v1.0 GPT-4o-mini run, old-run backups, `.bak` notebooks) has been removed;
see §2 for why v1.0 was replaced.
