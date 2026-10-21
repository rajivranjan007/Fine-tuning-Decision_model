"""Convert xLAM function-calling 60k -> Unsloth decision-model JSONL (tool selection).

Each row: user query + 3-5 candidate tools -> pick the one correct tool.
Usage:  python prepare_xlam_data.py
Source: https://huggingface.co/datasets/lockon/xlam-function-calling-60k (ungated mirror of Salesforce/xlam-function-calling-60k)
Output: data/xlam_train.jsonl, data/xlam_test.jsonl
"""
import json, os, random
from datasets import load_dataset

SEED, TEST_N, MIN_TOOLS, MAX_TOOLS = 3407, 1000, 3, 5
INSTR = "Which tool should be called to answer this request?"
rng = random.Random(SEED)

items, seen = [], set()
for r in load_dataset("lockon/xlam-function-calling-60k", split="train"):
    tools, answers = json.loads(r["tools"]), json.loads(r["answers"])
    gold = {a["name"] for a in answers}
    names = [t["name"] for t in tools]
    q = " ".join(r["query"].split())
    if len(gold) != 1 or not MIN_TOOLS <= len(tools) <= MAX_TOOLS: continue   # one correct tool, real choice
    if len(set(names)) != len(names) or q.lower() in seen: continue           # unique tool names / queries
    seen.add(q.lower())
    rng.shuffle(tools)                                                        # no position bias
    crit = {t["name"]: (t.get("description", "")[:200] + " | params: " + ", ".join(t.get("parameters", {}))) for t in tools}
    lab = gold.pop()
    items.append({
        "id": f"xlam_{r['id']}",
        "state": json.dumps({"message": q}),
        "questions": json.dumps({"tool": {"type": "choice", "instructions": INSTR, "criteria": crit}}),
        "gold": json.dumps({"tool": {"label": lab, "confidence": 1.0,
                                     "probabilities": {n: float(n == lab) for n in crit}}}),
    })

rng.shuffle(items)
os.makedirs("data", exist_ok=True)
for name, part in (("test", items[:TEST_N]), ("train", items[TEST_N:])):
    with open(f"data/xlam_{name}.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in part)
    print(name, len(part))
