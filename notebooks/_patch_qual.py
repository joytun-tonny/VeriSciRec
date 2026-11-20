# -*- coding: utf-8 -*-
import json, sys, ast
sys.stdout.reconfigure(encoding="utf-8")

def newcode(src, cid):
    return {"cell_type": "code", "id": cid, "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}
def newmd(src, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

def insert_after(nb, pred, cells):
    for i, c in enumerate(nb["cells"]):
        s = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        if pred(s):
            nb["cells"][i+1:i+1] = cells
            return True
    return False

# ============================ nb07: LIVE demo =================================
P7 = "07_maper_orchestration.ipynb"
nb7 = json.load(open(P7, encoding="utf-8"))

md7 = r'''## 5c. Curated qualitative example — LIVE relational generation + CiteCheck

Generates a **relation-eliciting** rationale over the retrieved candidates (so it contains
inter-paper claims), then CiteCheck verifies each `(A, relation, B)` and drops the flagged
relational hallucinations. This is the end-to-end system showcase. 💲 gated by `RUN_QUAL_DEMO`.'''

code7 = r'''# Relation-eliciting generation over the top-k candidates (references papers by title),
# so the rationale contains inter-paper claims for CiteCheck to verify.
def generate_relational_rationale(query, candidates, k=5):
    cs = candidates[:k]
    block = "\n".join(f'[{i+1}] "{c["title"]}": {c.get("abstract","")[:300]}' for i, c in enumerate(cs))
    instr = ("In 3-5 sentences, explain how the recommended papers relate TO EACH OTHER and to the "
             "query. Where supported, state explicit relations using cues like 'builds on'/'extends', "
             "'outperforms'/'compared to', 'uses the method of', 'supersedes'. Refer to each paper by "
             "its exact title in quotes. Ground every claim in the abstracts; do NOT fabricate relationships.")
    prompt = f"{instr}\n\nQuery: {query}\n\nPapers:\n{block}\n\nRelational rationale:"
    body = {"model": GEN_MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 250, "temperature": 0.3,
            "provider": {"order": [OPENROUTER_PROVIDER], "allow_fallbacks": False}}
    r = requests.post(f"{OPENROUTER_BASE}/chat/completions",
                      headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"}, json=body, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


# set True to spend ~1 OpenRouter call and produce the paper's qualitative example
RUN_QUAL_DEMO = False
if RUN_QUAL_DEMO and OPENROUTER_API_KEY:
    demo_q = "retrieval-augmented generation for reducing hallucination in LLMs"
    st = make_initial_state(test_user_id, demo_q, query_type="hybrid", max_iterations=1)
    st = node_profile_agent(st); st = node_retrieval_agent(st)
    cands = st["candidates"][:5]
    citecheck.ensure([c["paper_id"] for c in cands])
    rat = generate_relational_rationale(demo_q, cands)
    print("QUERY:", demo_q)
    print("\nRETRIEVED CANDIDATES:")
    for i, c in enumerate(cands):
        print(f"  [{i+1}] {c['paper_id']}  {c['title'][:60]}")
    print("\nGENERATED RELATIONAL RATIONALE:\n", rat, "\n")
    print("CiteCheck verification of each relational claim:")
    print("-" * 72)
    flagged = []
    for sent in _sentences(rat):
        det = detect_relation(sent, cands)
        if not det:
            continue
        a = det["paper_a"] or cands[0]["paper_id"]; b = det["paper_b"]
        verdict = citecheck.verify_relation(str(a), det["relation"], str(b)) if b else "unverifiable"
        mark = "  <- relational hallucination (flagged)" if verdict == "unsupported" else ""
        print(f"  [{det['relation']}] {a} -> {b}: {verdict.upper()}{mark}")
        print(f"     claim: {sent[:120]}")
        if verdict == "unsupported":
            flagged.append(sent)
    kept = [s for s in _sentences(rat) if s not in flagged]
    print("-" * 72)
    print("\nVERIFIED RATIONALE (relational hallucinations removed):\n", " ".join(kept))
    json.dump({"query": demo_q,
               "candidates": [{"paper_id": c["paper_id"], "title": c["title"]} for c in cands],
               "rationale": rat, "flagged_relational": flagged},
              open(OUT_DIR / "qual_example.json", "w", encoding="utf-8"))
    print("\nsaved ->", OUT_DIR / "qual_example.json")
else:
    print("RUN_QUAL_DEMO=False — set True (and have an OpenRouter key) to generate the curated example.")'''

assert insert_after(nb7,
    lambda s: 'MAPER Recommendations for' in s and 'final_recommendations' in s,
    [newmd(md7, "qual-live-md"), newcode(code7, "qual-live-code")]), "nb07 anchor not found"
json.dump(nb7, open(P7, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ============================ nb02: benchmark example =========================
P2 = "02_relational_finding.ipynb"
nb2 = json.load(open(P2, encoding="utf-8"))

md2 = r'''## 5c. Worked qualitative example — the blind spot (benchmark + CiteCheck), free

A single relational claim a **text verifier passed** but **CiteCheck flagged** (`unsupported`),
shown with both verdicts and the human gold. This is the evidence figure for the finding.'''

code2 = r'''# Find a relational claim a TEXT verifier did NOT flag but CiteCheck flagged unsupported.
_ex = None
for _inst in bench:
    for _lbl in _inst.get("hallucination_labels", []):
        _rel = _lbl.get("relational")
        if not (_rel and _rel.get("paper_a") and _rel.get("paper_b")):
            continue
        _cv = citecheck.get(f"{_inst['id']}|{_lbl['claim_id']}")
        if _cv == "unsupported" and _lbl.get("verdict") in ("supported", "neutral", "general_statement"):
            _ex = (_inst, _lbl, _cv); break
    if _ex:
        break

if _ex:
    _inst, _lbl, _cv = _ex; _rel = _lbl["relational"]
    print("Instance:", _inst["id"], "| domain:", _inst.get("domain"))
    print("\nReference rationale (excerpt):\n", (_inst.get("reference_rationale", "") or "")[:600])
    print("\nRelational claim under test:")
    print("  claim    :", _lbl["claim_text"][:170])
    print(f"  relation : ({_rel['paper_a']}) {_rel['relation']} ({_rel['paper_b']})")
    print("  text verifier (MiniCheck/NLI):", _lbl.get("verdict"))
    print("  CiteCheck (citation graph)   :", _cv, "  <- catches what the text verifier missed")
    print("  human gold                   :", _lbl.get("human_label") or "[pending annotation]")
else:
    print("No blind-spot example yet - need citecheck_results.json + a relational claim (run 01_citecheck).")'''

assert insert_after(nb2,
    lambda s: 'CiteCheck head-to-head on the relational gold slice' in s,
    [newmd(md2, "qual-bench-md"), newcode(code2, "qual-bench-code")]), "nb02 anchor not found"
json.dump(nb2, open(P2, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ---- syntax check both ----
for tag, nb in (("nb07", nb7), ("nb02", nb2)):
    bad = 0
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            try: ast.parse(src)
            except SyntaxError as e:
                bad += 1; print(f"SYNTAX {tag} {c.get('id')}: {e}")
    print(f"{tag}: {'syntax-clean' if bad == 0 else f'{bad} errors'}")
print("qual-example cells inserted.")
