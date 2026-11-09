"""Thinking off with majority voting at matched tokens (CPU, existing outputs).

maj@j over the four samples of a problem: every subset of j samples votes (each sample's answer under the authors'
scoring: a correct sample votes for the reference, a wrong one for its extracted answer, one without an answer does
not vote); the most common answer wins, ties are split evenly (expected accuracy), and a subset with no vote is wrong.
Values average all C(4, j) subsets, so maj@1 is the accuracy of Table 1. Tokens: the mean generated tokens of j samples
per problem. The agent's single sample is compared with thinking off's maj@j at the largest j whose tokens stay within
the edit's; 95% intervals by paired bootstrap over problems.

    python -m wsa.majk        (writes results/majk.json)
"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from wsa.grade import extract, mc_extract

RUNS = Path("results/runs")
ARMS = {"MATH-500": {"agent's edit": "edit_authors/think35_math500", "thinking off, full": "e3/test/nothink35_32k_math500",
                     "thinking off, 8k": "base/nothink35_math500", "base": "base/think35_math500"},
        "GSM8K": {"agent's edit": "edit_authors/think35_gsm8k", "thinking off, full": "e3/test/nothink35_32k_gsm8k",
                  "thinking off, 8k": "base/nothink35_gsm8k", "base": "base/think35_gsm8k"},
        # multiple choice: a wrong sample votes for its extracted option (wsa.grade.mc_extract)
        "MMLU-Pro math": {"agent's edit": "headroom/edit/think35_mmlupro_math",
                          "thinking off, full": "headroom/fix/nothink35_32k_mmlupro_math",
                          "thinking off, 8k": "headroom/fix/nothink35_mmlupro_math", "base": "headroom/base/think35_mmlupro_math"}}
MC = {"MMLU-Pro math"}


def votes(stem, mc=False):
    corr = {r["id"]: r["correct"] for r in json.load(open(RUNS / f"{stem}.per_problem.json"))}
    v, tok = {}, {}
    for line in open(RUNS / f"{stem}.jsonl"):
        r = json.loads(line)
        if r["id"] not in corr or r["k"] >= 4:
            continue
        if corr[r["id"]][r["k"]]:
            a = "<correct>"
        else:
            a = mc_extract(r["text"]) if mc else extract(r["text"])[0]
            a = None if a is None else str(a).replace(" ", "")
        v.setdefault(r["id"], {})[r["k"]] = a
        tok.setdefault(r["id"], {})[r["k"]] = r["n_tok"]
    return v, tok


def maj(answers):
    c = Counter(a for a in answers if a is not None)
    if not c:
        return 0.0
    top = max(c.values())
    win = [a for a, n in c.items() if n == top]
    return sum(a == "<correct>" for a in win) / len(win)


def main():
    out = {}
    for ds, arms in ARMS.items():
        V = {name: votes(stem, ds in MC) for name, stem in arms.items()}
        probs = sorted(set.intersection(*(set(v) for v, _ in V.values())))
        R = {"problems": len(probs), "arms": {}}
        acc = {}
        for name, (v, tok) in V.items():
            acc[name] = {}
            R["arms"][name] = {}
            for j in range(1, 5):
                a = np.array([np.mean([maj([v[p][k] for k in s]) for s in combinations(range(4), j)]) for p in probs])
                t = float(np.mean([j * np.mean(list(tok[p].values())) for p in probs]))
                acc[name][j] = a
                R["arms"][name][j] = {"accuracy": float(a.mean()), "tokens": t}
        e1, et = acc["agent's edit"][1], R["arms"]["agent's edit"][1]["tokens"]
        idx = np.random.default_rng(0).integers(0, len(probs), (10000, len(probs)))
        R["edit_minus_thinking_off_majj"] = {}
        for j in range(1, 5):
            d = e1 - acc["thinking off, full"][j]
            bs = d[idx].mean(1)
            R["edit_minus_thinking_off_majj"][j] = {
                "diff": [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "thinking_off_tokens": R["arms"]["thinking off, full"][j]["tokens"], "edit_tokens": et}
        out[ds] = R
        print(ds, f"{len(probs)} problems")
        for name, r in R["arms"].items():
            print(f"   {name:20s}", "  ".join(f"maj@{j} {r[j]['accuracy']*100:.1f} ({r[j]['tokens']/1000:.1f}k)" for j in range(1, 5)))
        for j, r in R["edit_minus_thinking_off_majj"].items():
            d = r["diff"]
            print(f"   edit@1 ({et/1000:.1f}k) - thinking off maj@{j} ({r['thinking_off_tokens']/1000:.1f}k): "
                  f"{d[0]*100:+.1f} [{d[1]*100:.1f}, {d[2]*100:.1f}]")
    Path("results/majk.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
