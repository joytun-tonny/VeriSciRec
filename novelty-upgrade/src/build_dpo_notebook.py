"""
Generates novelty-upgrade/notebooks/09_dpo_training.ipynb.

Why generate rather than hand-write the .ipynb JSON: easier to keep in sync with
graph_utils.py, easier to review as plain Python, and avoids hand-authoring nbformat
JSON. Every code cell is compile-checked at the bottom of this script before writing
the notebook out, so a broken cell fails loudly here instead of silently on Kaggle.

This notebook is meant to run on Kaggle GPU (the QLoRA/DPO cell needs real VRAM for a
70B model in 4-bit) -- everything before that cell is CPU-only and can also be run
locally to sanity-check the preference-pair construction before uploading.
"""
import ast
import json
from pathlib import Path


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


cells = []

cells.append(md(
    "# 09 - Graph-Verified DPO Training (novelty item 9)\n"
    "\n"
    "Built fresh in `novelty-upgrade/`, reusing the validated preference-pair "
    "construction approach from `../notebooks/08_dpo_pipeline.ipynb` but rebuilt "
    "against `novelty-upgrade/src/graph_utils.py` (this session's corrected, "
    "validated reimplementation of the decision rule -- see that module's "
    "`_self_test` for why it was corrected).\n"
    "\n"
    "**Run on Kaggle** (GPU required for the training cell). Sections 1-2 are "
    "CPU-only and safe to run locally first to sanity-check the pair counts before "
    "uploading."
))

cells.append(md("## 1. Setup"))

cells.append(code(
    "# == Environment + paths =========================================================\n"
    "import json, os, re, sys, time\n"
    "from pathlib import Path\n"
    "\n"
    "# Kaggle detection uses an env var Kaggle actually sets, not path existence: on\n"
    "# Windows, Path('/kaggle/working').exists() is NOT a reliable off-Kaggle check --\n"
    "# pathlib treats a leading '/' as relative to the current drive's root, so if the\n"
    "# machine happens to have an unrelated C:\\kaggle or F:\\kaggle directory for any\n"
    "# other reason, that check silently returns True. (Hit exactly this locally: a\n"
    "# pre-existing, totally unrelated F:\\Kaggle directory made an earlier version of\n"
    "# this cell write output there instead of ../outputs.)\n"
    "ON_KAGGLE = bool(os.environ.get(\"KAGGLE_KERNEL_RUN_TYPE\") or os.environ.get(\"KAGGLE_URL_BASE\"))\n"
    "\n"
    "def _find(name, prefer_largest=True):\n"
    "    \"\"\"Search for `name`; if multiple matches exist, prefer the largest file --\n"
    "    a complete/authoritative cache is virtually always bigger than a stale or\n"
    "    partial one left over from an earlier run. (Also hit this locally: a 650KB\n"
    "    stale citecheck_refs_cache.json under notebooks/outputs/ vs. the real 43MB\n"
    "    one under benchmarks/final/relational/ -- naive first-match picked the wrong\n"
    "    one and silently produced a near-empty citation graph.)\"\"\"\n"
    "    roots = [\"/kaggle/input\"] if ON_KAGGLE else [\"..\", \"../..\", \".\"]\n"
    "    hits = []\n"
    "    for root in roots:\n"
    "        r = Path(root)\n"
    "        if r.exists():\n"
    "            hits.extend(r.rglob(name))\n"
    "    if not hits:\n"
    "        return None\n"
    "    if prefer_largest:\n"
    "        hits.sort(key=lambda p: p.stat().st_size, reverse=True)\n"
    "    return hits[0]\n"
    "\n"
    "# novelty-upgrade/src on the path, so we can reuse graph_utils.py -- searched by\n"
    "# content (contains graph_utils.py), not a fixed relative path: on Kaggle, an\n"
    "# attached dataset lands under /kaggle/input/<dataset-name>/..., where neither\n"
    "# '../src' nor 'src' exists.\n"
    "def _find_src_dir():\n"
    "    for c in (Path(\"../src\"), Path(\"src\"), Path(\"/kaggle/working/src\")):\n"
    "        if (c / \"graph_utils.py\").exists():\n"
    "            return c\n"
    "    if ON_KAGGLE:\n"
    "        hits = list(Path(\"/kaggle/input\").rglob(\"graph_utils.py\"))\n"
    "        if hits:\n"
    "            return hits[0].parent\n"
    "    return Path(\"src\")\n"
    "\n"
    "SRC_DIR = _find_src_dir()\n"
    "sys.path.insert(0, str(SRC_DIR))\n"
    "from graph_utils import CitationGraph, load_relational_claims, verify\n"
    "\n"
    "BENCH = _find(\"scireqa_full_relational_v1.1_humanmerged.jsonl\") or _find(\"scireqa_full_relational_v1.1.jsonl\")\n"
    "REFS_CACHE = _find(\"citecheck_refs_cache.json\")\n"
    "assert BENCH and REFS_CACHE, \"attach/locate the benchmark JSONL and citecheck_refs_cache.json\"\n"
    "\n"
    "OUT_DIR = Path(\"/kaggle/working\") if ON_KAGGLE else Path(\"../outputs\")\n"
    "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "print(\"ON_KAGGLE:\", ON_KAGGLE)\n"
    "print(\"SRC_DIR:\", SRC_DIR.resolve() if SRC_DIR.exists() else f\"{SRC_DIR} (NOT FOUND -- attach the src/ dataset)\")\n"
    "print(\"BENCH:\", BENCH, f\"({BENCH.stat().st_size:,} bytes)\")\n"
    "print(\"REFS_CACHE:\", REFS_CACHE, f\"({REFS_CACHE.stat().st_size:,} bytes)\")\n"
    "print(\"OUT_DIR:\", OUT_DIR)\n"
))

cells.append(md(
    "## 2. Graph-verified preference pairs\n"
    "\n"
    "Same construction logic as the original notebook (prune sentences "
    "CiteCheck/SciRecTrace flags as unsupported relational claims to build "
    "`chosen`; the untouched original rationale is `rejected`), rebuilt here "
    "against the corrected `verify()` so pair labels are consistent with "
    "everything else built this session (counterfactual harness, hard negatives)."
))

cells.append(code(
    "# == Build graph-grounded preference pairs, using this session's verify() =========\n"
    "import re as _re\n"
    "\n"
    "def _sentences(text):\n"
    "    return [s.strip() for s in _re.split(r\"(?<=[.!?])\\s+\", str(text).strip()) if s.strip()]\n"
    "\n"
    "graph = CitationGraph.load(REFS_CACHE)\n"
    "bench = [json.loads(l) for l in open(BENCH, encoding=\"utf-8\") if l.strip()]\n"
    "print(f\"benchmark instances: {len(bench)} | graph nodes: {len(graph)}\")\n"
    "\n"
    "preference_pairs = []\n"
    "for inst in bench:\n"
    "    rationale = (inst.get(\"reference_rationale\") or \"\").strip()\n"
    "    labels = inst.get(\"hallucination_labels\", [])\n"
    "    if not rationale or not labels:\n"
    "        continue\n"
    "\n"
    "    flagged_texts = set()\n"
    "    for lbl in labels:\n"
    "        rel = lbl.get(\"relational\")\n"
    "        if not rel or not rel.get(\"is_relational\"):\n"
    "            continue\n"
    "        a, rho, b = rel.get(\"paper_a\"), rel.get(\"relation\"), rel.get(\"paper_b\")\n"
    "        if not (a and rho and b):\n"
    "            continue\n"
    "        verdict = verify(a, rho, b, graph)\n"
    "        if verdict == \"unsupported\":\n"
    "            flagged_texts.add((lbl.get(\"claim_text\") or \"\").strip())\n"
    "\n"
    "    if not flagged_texts:\n"
    "        continue\n"
    "    kept = [s for s in _sentences(rationale) if s not in flagged_texts]\n"
    "    if not kept:\n"
    "        continue\n"
    "    chosen = \" \".join(kept)\n"
    "    rejected = rationale\n"
    "    if chosen.strip() == rejected.strip():\n"
    "        continue\n"
    "\n"
    "    query_text = str(inst.get(\"query\", \"\"))[:200]\n"
    "    prompt = f\"Recommend papers for the query and explain why, grounded in the abstracts.\\nQuery: {query_text}\"\n"
    "    preference_pairs.append({\n"
    "        \"instance_id\": inst[\"id\"], \"prompt\": prompt, \"chosen\": chosen, \"rejected\": rejected,\n"
    "        \"n_flagged_relational\": len(flagged_texts), \"domain\": inst.get(\"domain\", \"\"),\n"
    "    })\n"
    "\n"
    "print(f\"Built {len(preference_pairs)} graph-grounded pairs\")\n"
    "\n"
    "filtered_pairs = [p for p in preference_pairs\n"
    "                  if p[\"chosen\"] != p[\"rejected\"] and len(p[\"chosen\"]) > 30 and len(p[\"rejected\"]) > 30]\n"
    "print(f\"After filtering: {len(filtered_pairs)} pairs\")\n"
    "\n"
    "pairs_path = OUT_DIR / \"preference_pairs.jsonl\"\n"
    "with open(pairs_path, \"w\", encoding=\"utf-8\") as f:\n"
    "    for p in filtered_pairs:\n"
    "        f.write(json.dumps(p, ensure_ascii=False) + \"\\n\")\n"
    "print(\"Saved:\", pairs_path)\n"
))

cells.append(md(
    "## 3. QLoRA DPO training (Kaggle GPU required)\n"
    "\n"
    "Same package/version check pattern as the original notebook -- installs only "
    "what's missing, no version pins (the earlier lesson documented there: pinning "
    "old versions against a modern Kaggle image breaks numpy/bitsandbytes)."
))

cells.append(code(
    "# == Check + INSTALL required packages (minimal, non-destructive) ================\n"
    "import subprocess, sys\n"
    "\n"
    "required = [\"transformers\", \"peft\", \"trl\", \"bitsandbytes\", \"datasets\"]\n"
    "missing = []\n"
    "for pkg in required:\n"
    "    try:\n"
    "        mod = __import__(pkg)\n"
    "        print(f\"  already present: {pkg} == {getattr(mod, '__version__', '?')}\")\n"
    "    except ImportError:\n"
    "        missing.append(pkg)\n"
    "        print(f\"  missing: {pkg}\")\n"
    "\n"
    "if missing:\n"
    "    print(f\"\\nInstalling ONLY what's missing, no version pins: {missing}\")\n"
    "    subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\"] + missing, check=True)\n"
    "else:\n"
    "    print(\"\\nNothing missing.\")\n"
))

cells.append(code(
    "# == DPO training ==================================================================\n"
    "import torch\n"
    "from datasets import Dataset\n"
    "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n"
    "from peft import LoraConfig, get_peft_model, TaskType\n"
    "from trl import DPOConfig, DPOTrainer\n"
    "\n"
    "MODEL_NAME = \"meta-llama/Llama-3.3-70B-Instruct\"\n"
    "# Under OUT_DIR, not a bare relative path: Path('..').exists() is true on almost\n"
    "# any filesystem (there's almost always a parent directory), so a naive\n"
    "# '../lora-adapter if Path(\"..\").exists() else ...' check never actually falls\n"
    "# back -- and on Kaggle, '..' from /kaggle/working is /kaggle, which is not\n"
    "# writable (only /kaggle/working is). OUT_DIR is already resolved correctly for\n"
    "# both environments above.\n"
    "ADAPTER_OUT = OUT_DIR / \"lora-adapter\"\n"
    "\n"
    "assert torch.cuda.is_available(), \"This cell needs a GPU runtime (Kaggle: enable GPU accelerator).\"\n"
    "\n"
    "bnb_config = BitsAndBytesConfig(\n"
    "    load_in_4bit=True, bnb_4bit_quant_type=\"nf4\",\n"
    "    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,\n"
    ")\n"
    "\n"
    "tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)\n"
    "if tokenizer.pad_token is None:\n"
    "    tokenizer.pad_token = tokenizer.eos_token\n"
    "\n"
    "model = AutoModelForCausalLM.from_pretrained(\n"
    "    MODEL_NAME, quantization_config=bnb_config, device_map=\"auto\", torch_dtype=torch.bfloat16,\n"
    ")\n"
    "\n"
    "lora_config = LoraConfig(\n"
    "    r=16, lora_alpha=32, lora_dropout=0.05, bias=\"none\", task_type=TaskType.CAUSAL_LM,\n"
    "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],\n"
    ")\n"
    "model = get_peft_model(model, lora_config)\n"
    "model.print_trainable_parameters()\n"
    "\n"
    "pairs = [json.loads(l) for l in open(OUT_DIR / \"preference_pairs.jsonl\", encoding=\"utf-8\")]\n"
    "dpo_dataset = Dataset.from_list([\n"
    "    {\"prompt\": p[\"prompt\"], \"chosen\": p[\"chosen\"], \"rejected\": p[\"rejected\"]} for p in pairs\n"
    "])\n"
    "print(f\"DPO training set: {len(dpo_dataset)} pairs\")\n"
    "\n"
    "dpo_config = DPOConfig(\n"
    "    output_dir=str(ADAPTER_OUT), per_device_train_batch_size=1, gradient_accumulation_steps=8,\n"
    "    num_train_epochs=1, learning_rate=5e-6, beta=0.1, logging_steps=10,\n"
    "    save_strategy=\"epoch\", bf16=True, remove_unused_columns=False,\n"
    ")\n"
    "\n"
    "# trl renamed DPOTrainer's tokenizer= kwarg to processing_class= in newer releases;\n"
    "# since the install step above deliberately takes whatever trl version pip resolves\n"
    "# to (no pins -- pinning old versions against a modern Kaggle image breaks\n"
    "# numpy/bitsandbytes), detect which name this installed version actually accepts\n"
    "# instead of hardcoding one and risking a TypeError on the training cell.\n"
    "import inspect\n"
    "_dpo_kwargs = dict(model=model, args=dpo_config, train_dataset=dpo_dataset)\n"
    "_dpo_params = inspect.signature(DPOTrainer.__init__).parameters\n"
    "_dpo_kwargs[\"processing_class\" if \"processing_class\" in _dpo_params else \"tokenizer\"] = tokenizer\n"
    "trainer = DPOTrainer(**_dpo_kwargs)\n"
    "trainer.train()\n"
    "trainer.save_model(str(ADAPTER_OUT))\n"
    "print(\"Saved adapter to\", ADAPTER_OUT)\n"
))

cells.append(md(
    "## 4. Evaluation: does DPO actually reduce the relational-hallucination rate?\n"
    "\n"
    "Regenerate rationales for the SAME 68 human-verified instances the paper's "
    "ablation table (Table 6) already uses, with the DPO-tuned model instead of the "
    "base model, then run them through the identical decompose to verify pipeline "
    "and compare the pre-exclusion relational-hallucination rate against the "
    "existing (a)/(b)/(c) rows. This is what turns \"we built preference pairs\" "
    "into an actual, reportable (d) row."
))

cells.append(code(
    "# == Load the tuned model and generate on the 68-instance human-verified checkpoint ==\n"
    "# NOTE: fill in once the human-verified instance IDs/prompts are available in this\n"
    "# session -- this cell intentionally left as a template, not run blind, so the\n"
    "# comparison is against the EXACT same 68 instances Table 6 already reports on, not\n"
    "# a different sample that would make the comparison invalid.\n"
    "\n"
    "from peft import PeftModel\n"
    "\n"
    "base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config,\n"
    "                                             device_map=\"auto\", torch_dtype=torch.bfloat16)\n"
    "tuned = PeftModel.from_pretrained(base, str(ADAPTER_OUT))\n"
    "tuned.eval()\n"
    "\n"
    "def generate(prompt: str, max_new_tokens: int = 180) -> str:\n"
    "    inputs = tokenizer(prompt, return_tensors=\"pt\").to(tuned.device)\n"
    "    with torch.no_grad():\n"
    "        out = tuned.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2, do_sample=True)\n"
    "    return tokenizer.decode(out[0][inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True)\n"
    "\n"
    "# TODO once run: load the 68-instance checkpoint's prompts, generate, decompose each\n"
    "# rationale into claims (reuse the same decomposition logic as ../notebooks/06),\n"
    "# verify relational claims with graph_utils.verify(), and compute the pre-exclusion\n"
    "# relational-hallucination rate exactly as Table 6's (a)/(b)/(c) rows do, so\n"
    "# (d) DPO is directly comparable.\n"
    "print(\"Template ready -- fill in the 68-instance prompt list before running for real.\")\n"
))

# -- Validate every code cell compiles before writing the notebook out --------------
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise SystemExit(f"Cell {i} has a syntax error, not writing notebook: {e}\n---\n{src}")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path(__file__).resolve().parents[1] / "notebooks" / "09_dpo_training.ipynb"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out_path} ({len(cells)} cells, all code cells compile-checked)")
