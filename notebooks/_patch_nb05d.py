# -*- coding: utf-8 -*-
import json, sys, ast
sys.stdout.reconfigure(encoding="utf-8")
P = "05_hybrid_retrieval_agent.ipynb"
nb = json.load(open(P, encoding="utf-8"))

old = '_up = _find("user_profiles.json"); user_profiles = json.load(open(_up, encoding="utf-8")) if _up else {}'
new = ('_UP_HARD = Path("/kaggle/input/datasets/joytuntonny/rationale-subset/user_profiles.json")\n'
       '_up = _UP_HARD if _UP_HARD.exists() else _find("user_profiles.json")\n'
       'user_profiles = json.load(open(_up, encoding="utf-8")) if _up else {}')

found = False
for c in nb["cells"]:
    if c.get("id") == "ad530305":
        s = c["source"]
        assert s.count(old) == 1, f"found {s.count(old)}"
        c["source"] = s.replace(old, new); found = True; break
assert found

json.dump(nb, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
bad = 0
for c in nb["cells"]:
    if c["cell_type"] == "code":
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        try: ast.parse(src)
        except SyntaxError as e:
            bad += 1; print("SYNTAX ERROR", c.get("id"), e)
print("nb05 user_profiles path pinned |", "syntax-clean" if bad == 0 else f"{bad} errors")
