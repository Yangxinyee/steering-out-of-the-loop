"""Prompt-only arm: Qwen3.5-0.8B unchanged, think35 plus one instruction. The wording was
picked on practice (sel200, 1 sample); the pick answers MATH-500, GSM8K and MedQA test, 4 samples each. Every arm is
set against the base model on the same samples, by two rules, with the other fixes as reference rows.

  official  the authors' reading of every answer, a cut one included
  strict    a cut answer is wrong
  rescued   same problem and sample index: cut under the base model, finished under the arm (as in wsa.rescue)
  looping   cut answers whose end repeats (the authors' rule, wsa.grade.looping)

    python -m wsa.prompt_only --runs results/runs --out results/prompt_only
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wsa.grade import TAIL_CHARS, looping
from wsa.rescue import boot_ci, load

WORDINGS = {"think35_pr1": "one sentence after the problem: stop and answer on noticing repetition", "think35_pr2": "the same sentence as a system message",
            "think35_pr3": "a firmer sentence after the problem: do not repeat; on going over the same steps, stop thinking and answer"}
ARMS = {"prompt": "prompt only (pr2)", "authors_edit": "agent's edit", "t1": "T=1.0 (model card)", "bf8k": "budget forcing 8k",
        "nothink": "thinking off"}
SETS = {   # name: (kind, base, {arm: stem})
    "MATH-500": ("math", "base/think35_math500", {
        "prompt": "prompt/think35_pr2_math500", "authors_edit": "edit_authors/think35_math500",
        "t1": "base/card_think_math500", "bf8k": "base/think35_bf8192_math500", "nothink": "base/nothink35_math500"}),
    "GSM8K": ("math", "base/think35_gsm8k", {
        "prompt": "prompt/think35_pr2_gsm8k", "authors_edit": "edit_authors/think35_gsm8k",
        "t1": "base/card_think_gsm8k", "bf8k": "base/think35_bf8192_gsm8k", "nothink": "base/nothink35_gsm8k"}),
    "MedQA": ("mc", "med_base/think35_medqa", {
        "prompt": "prompt/think35_pr2_medqa", "authors_edit": "med_edit/think35_medqa",
        "t1": "med_base/card_think_medqa", "bf8k": "med_base/think35_bf8192_medqa", "nothink": "med_base/nothink35_medqa"}),
}


def loop_share(runs, stem):
    n = loops = 0
    for line in open(runs / f"{stem}.jsonl"):
        g = json.loads(line)
        if g["finish"] == "length":
            n += 1
            loops += looping(g["text"][-TAIL_CHARS:])
    return loops / n if n else None


def per_problem(d, strict):
    q = defaultdict(list)
    for (pid, _), (c, fin, _) in d.items():
        q[pid].append(c and (fin or not strict))
    return {p: float(np.mean(v)) for p, v in q.items()}


def describe(d):
    v = list(d.values())
    return {"official": float(np.mean([c for c, _, _ in v])), "strict": float(np.mean([c and f for c, f, _ in v])),
            "cut": float(np.mean([not f for _, f, _ in v])), "n": len(v)}


def main(a):
    runs = Path(a.runs)
    res = {"practice_pick": {}, "sets": {}}
    # the base model on the same 200 practice problems: the sel200 rows of its 500-problem run
    sel200 = {json.loads(l)["id"] for l in open(a.sel200)}
    s = describe({k: v for k, v in load(runs, "base/think35_sel", "math").items() if k[0] in sel200})
    res["practice_pick"]["think35"] = {"wording": "base model, no added instruction", "acc": s["official"], "cut": s["cut"], "n": s["n"]}
    for w, text in WORDINGS.items():
        s = json.load(open(runs / "prompt" / f"{w}_sel200.scores.json"))
        res["practice_pick"][w] = {"wording": text, "acc": s["acc"], "cut": s["cut_share"], "n": s["total"]}
    for name, (kind, base_stem, arms) in SETS.items():
        base = load(runs, base_stem, kind)
        S = {"base": {**describe(base), "cut_looping": loop_share(runs, base_stem)}, "arms": {}}
        for arm, stem in arms.items():
            if not (runs / f"{stem}.per_problem.json").exists():
                continue
            d = load(runs, stem, kind)
            keys = [k for k in base if k in d]
            row = describe({k: d[k] for k in keys})
            for rule, strict in (("official", False), ("strict", True)):
                pa, pb = per_problem({k: d[k] for k in keys}, strict), per_problem({k: base[k] for k in keys}, strict)
                m, lo, hi = boot_ci([pa[p] - pb[p] for p in pa])
                row[f"gain_{rule}"] = {"gain": m, "lo": lo, "hi": hi}
            resc = [k for k in keys if not base[k][1] and d[k][1]]
            row["rescued"] = len(resc)
            row["rescued_acc"] = float(np.mean([d[k][0] for k in resc])) if resc else None
            row["newly_cut"] = sum(base[k][1] and not d[k][1] for k in keys)
            if arm == "prompt":
                row["cut_looping"] = loop_share(runs, stem)
            S["arms"][arm] = row
        res["sets"][name] = S
    Path(a.out + ".json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--sel200", default="results/data/sel200.jsonl")
    ap.add_argument("--out", default="results/prompt_only")
    main(ap.parse_args())
