# SciRecQA — Human Annotation Guidelines

## 1. What you are doing (and why)

Each **claim** is one sentence taken from an AI-generated *rationale* that explains why a set
of scientific papers were recommended for a query. Your job is to judge, for each claim:

1. **Is the claim grounded** — i.e., is it supported by the provided evidence (the recommended
   papers' abstracts) and true? → fill **`annotator_label`**
2. **If it is *not* grounded, what *kind* of error is it?** → fill **`annotator_type`**

These human judgments are the **gold standard**. We use them to measure how well automatic
verifiers detect hallucinations — especially **relational** ones (claims about relationships
*between* papers), which are the focus of this study.

You fill **only two columns**: `annotator_label` and `annotator_type`. Do not change anything else.

---

## 2. What you see in each row (CSV columns)

| Column | Meaning |
|---|---|
| `claim_text` | The sentence to judge |
| `evidence` | Abstracts of the recommended (ground-truth) papers — your main evidence |
| `claim_type` | Auto category (factual_candidate / general_statement) — reference only |
| `auto_verdict` | An automatic verdict — **for reference only; ignore it while deciding.** Judge independently. |
| **`annotator_label`** | ← you fill |
| **`annotator_type`** | ← you fill |

> **Note on the `relation` column:** the master data also carries an automatic relation-type
> guess (e.g. `builds_on`), but that column is **not included in your sheet**. It is withheld
> before distribution and only merged back in afterward, so that your `annotator_type` call on
> whether a claim is relational is not anchored on the system's own guess. Decide whether a
> claim states a genuine paper↔paper relationship from its content alone (Section 5) — there is
> no hint column to verify or ignore.

> **Independence rule:** do **not** look at `auto_verdict` to make your decision, and do **not**
> decide relational claims by checking citation databases (Google Scholar / Semantic Scholar
> "cited by"). Judge from the papers' **content** and your expertise. (Those automatic signals
> are exactly what your gold is being compared against — using them makes the comparison circular.)

---

## 3. `annotator_label` — is the claim grounded?

Choose **one**:

| Label | Use when… |
|---|---|
| **supported** | The claim is **true and backed by the evidence** (abstracts) / your knowledge. |
| **unsupported** | The claim is **false, fabricated, or contradicted** by the evidence (a hallucination). |
| **neutral** | The claim is plausible but **cannot be confirmed or refuted** from the evidence + your knowledge (insufficient information / only partially holds). |
| **general_statement** | The sentence is **generic prose with no checkable factual content** (e.g. "These papers are interesting and useful for researchers."). |

---

## 4. `annotator_type` — if not grounded, what kind of error?

Choose **one**.

> **Key rule for `relational`:** set `annotator_type = relational` for **any genuine
> paper↔paper relational claim — whether it is supported, unsupported, or neutral** — so the
> full relational slice can be identified. For every *other* type (entity/factual/relevance/
> reasoning) use the type **only when `annotator_label = unsupported`**; otherwise use `none`.

| Type | The hallucination is about… | Example of an *unsupported* claim |
|---|---|---|
| `none` | (not a hallucination) | — |
| `entity` | a wrong/invented **entity**: paper title, author, dataset, method name | "Paper [2] introduces the **ImageNet** benchmark" (it doesn't) |
| `factual` | a wrong **fact/number**: metric, year, result | "achieves **92%** F1" when it doesn't |
| `relevance` | a false claim that a paper **is relevant** to the query | "this paper directly addresses your query" when it's off-topic |
| `reasoning` | a flawed **inference**: the conclusion doesn't follow | "because both use transformers, [1] must outperform [2]" |
| **`relational`** | a wrong **relationship *between two papers***: builds-on / extends / uses / outperforms / supersedes | "Paper [1] **builds on** Paper [3]" when [1] does not build on [3] |

---

## 5. Relational claims — the focus (read carefully)

A **relational claim** asserts a relationship **between two specific papers** (or a paper and the
query paper), e.g.:
- *builds_on / extends / based-on*: "[1] builds on [3]'s method"
- *compares / outperforms / in-contrast*: "[2] outperforms [4]"
- *uses_method_of*: "[5] uses the approach from [2]"
- *supersedes / replaces*: "[4] supersedes [1]"

**How to judge it:**
1. Identify the **two papers** and the **claimed relationship**.
2. Decide whether that relationship **genuinely holds**, based on what the papers actually *do*
   (their methods/contributions) — from the abstracts, and by reading the papers if the abstracts
   are insufficient (inter-paper relations are often not stated in a single abstract).
3. Record — **always set `annotator_type = relational`** for a genuine relational claim, then set the label:
   - relationship clearly **holds** → `supported`, type **`relational`**.
   - relationship is **false / overstated / not real** → `unsupported`, type **`relational`**.
   - **cannot determine** from available information → `neutral`, type **`relational`**.
   (Marking `relational` on the true ones too is what lets us measure verifiers on the *whole* relational slice, not just the false ones.)

**Not a real paper↔paper relation?** If the sentence relates a paper to a *concept/topic* rather
than to another paper (e.g. "[1] builds on **the idea of** attention"), or merely says two papers
*share a topic* without a real dependency (e.g. "[1] and [2] **both focus on** segmentation"),
that is **not** a relational claim — judge it as a normal relevance/reasoning statement (and do
**not** use type `relational`), based on the claim's own wording, not any external signal.

**Independence (critical):** decide whether "[1] builds on [3]" is true by understanding the two
papers — **not** by checking whether [1] cites [3] in a citation database. Your content-based
judgment is the gold against which the citation-graph verifier is evaluated.

---

## 6. Step-by-step decision procedure (do this for every row)

1. Read `claim_text` and the `evidence`.
2. **Is it generic prose with nothing checkable?** → `general_statement`, type `none`. Done.
3. **Does it assert a relationship between two specific papers?**
   - **Yes** → set `annotator_type = relational`, then judge the relationship (Section 5): label `supported` / `unsupported` / `neutral`.
   - **No** → continue (this includes "both do X", shared-topic, and "builds on the *idea* of X" — type is **not** relational).
4. **Is the (atomic) claim true & backed by the evidence/your knowledge?**
   - Yes → `supported`, type `none`.
   - No (false/fabricated/contradicted) → `unsupported`, then pick the type: `entity` / `factual` / `relevance` / `reasoning`.
   - Can't tell → `neutral`, type `none`.

---

## 7. Tricky cases / FAQ

- **Mostly right, one wrong detail (e.g. wrong number):** → `unsupported`, type `factual` (or `entity`). A single fabricated fact makes the claim unsupported.
- **Vague but not wrong:** → `neutral`.
- **Claim uses dependency language ("builds on") but on closer reading it's really "both work on X":** → it's *not* relational; judge as relevance/reasoning, type accordingly (often `supported`/`neutral`).
- **Relation is plausible but you'd need to read the papers:** read the abstracts; if still unsure, **read the papers**; if truly undecidable, `neutral`.
- **Two errors in one sentence:** pick the **most central** error type.
- **The query paper is one of the two papers** ("in contrast to the query paper, [1]…"): that's still a valid relational claim — judge it the same way.

---

## 8. Consistency rules

- Judge **only** what the sentence claims — don't penalize it for things it didn't say.
- Be consistent: same situation → same label. When unsure between two labels, prefer `neutral`
  over guessing `supported`/`unsupported`.
- Spend extra care on **relational** claims — they are the study's focus and the hardest to judge.
- Do not collaborate on individual items during annotation (we measure inter-annotator agreement);
  disagreements are resolved later by majority vote.

---

## 9. Workflow (files)

1. You receive **`to_annotate_annotator_X.csv`** (X = your id: A, B, or C).
2. Fill **`annotator_label`** and **`annotator_type`** for **every row**. Leave all other columns unchanged.
3. Save your completed file as **`done_annotator_X.csv`** (e.g. `done_annotator_A.csv`) — same rows/order.
4. Return it for adjudication (the lead runs IAA + majority-vote and merges the gold labels).

**Allowed values (use exactly these strings):**
- `annotator_label`: `supported`, `unsupported`, `neutral`, `general_statement`
- `annotator_type`: `none`, `entity`, `factual`, `relevance`, `reasoning`, `relational`

---

## 10. Protocol & references (the methodology this annotation follows)

These guidelines are **task-specific** (every annotation effort writes its own), but the
**procedure** follows established corpus-annotation methodology so the labels are reproducible
and defensible — not ad-hoc.

### Procedure
1. **Iterative guideline development.** A pilot round (~50 claims) is annotated first; the
   guidelines are refined on ambiguous/disagreement cases; then the full study is run
   (the MAMA annotation cycle — Pustejovsky & Stubbs, 2012; Hovy & Lavid, 2010).
2. **≥3 independent expert annotators** (CS/NLP background); annotators do **not** confer on
   individual items.
3. **Inter-annotator agreement (IAA)** is computed and reported: Cohen's κ (pairwise),
   Fleiss' κ / Krippendorff's α (overall **and per type**), interpreted on the Landis & Koch
   (1977) scale, following Artstein & Poesio (2008).
4. **Adjudication** of disagreements by majority vote (ties flagged against the auto verdict).
5. **Transparency / release:** the guidelines, the raw per-annotator labels, and the adjudicated
   gold are released with a datasheet (Gebru et al., 2021) and data statement (Bender &
   Friedman, 2018).
6. **Report annotator qualifications** and the number of items / claims per annotator.

### Typology lineage (the error types are grounded, not invented)
- `entity` / `factual` / `reasoning` follow the factual-error typology of **FRANK**
  (Pagnoni et al., 2021), within the intrinsic/extrinsic faithfulness framing
  (Maynez et al., 2020; Ji et al., 2023).
- The **`relational`** type is grounded in scientific relation extraction (**SciERC** — Luan et
  al., 2018) and citation-intent classification (**SciCite** — Cohan et al., 2019).
- Annotation-of-hallucination precedents: **HaluEval** (Li et al., 2023), **RAGTruth**
  (Niu et al., 2024).

### References (verify exact venue/year before the .bib)
- Artstein, R. & Poesio, M. (2008). *Inter-Coder Agreement for Computational Linguistics.* Computational Linguistics 34(4).
- Cohen, J. (1960); Fleiss, J. L. (1971); Krippendorff, K. (2004) — agreement coefficients.
- Landis, J. R. & Koch, G. G. (1977) — κ interpretation scale.
- Pustejovsky, J. & Stubbs, A. (2012). *Natural Language Annotation for Machine Learning.* O'Reilly.
- Hovy, E. & Lavid, J. (2010). *Towards a 'Science' of Corpus Annotation.*
- Pagnoni, A. et al. (2021). *Understanding Factuality in Abstractive Summarization with FRANK.* NAACL.
- Maynez, J. et al. (2020). *On Faithfulness and Factuality in Abstractive Summarization.* ACL.
- Ji, Z. et al. (2023). *Survey of Hallucination in NLG.* ACM Computing Surveys.
- Luan, Y. et al. (2018). *SciERC.* EMNLP. · Cohan, A. et al. (2019). *SciCite.* NAACL.
- Li, J. et al. (2023). *HaluEval.* EMNLP. · Niu, C. et al. (2024). *RAGTruth.* ACL.
- Gebru, T. et al. (2021). *Datasheets for Datasets.* · Bender, E. & Friedman, B. (2018). *Data Statements for NLP.* · Bowman, S. & Dahl, G. (2021).
