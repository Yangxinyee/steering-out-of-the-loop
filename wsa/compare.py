"""Reconcile our reproduction with the authors' published per-problem scores (exp8_math/eval/<run>/parts/*).

Exact token-for-token agreement is not expected: different vLLM version (0.32 vs 0.33), different GPU, and per-request
seeds instead of one engine seed. What must agree is the result, per problem and in total. The rule, fixed before
any of our full results were seen:

  For each of MATH-500 and GSM8K, base and edited model: d = our accuracy - theirs, with a 95% interval from a paired
  bootstrap over problems (each problem's mean over its samples, 10,000 resamples). MISMATCH if the interval excludes
  0 AND |d| > 2 points. The same for the share of answers cut at 32,768 tokens (aggregate only; the authors publish it
  per shard, not per problem; interval from the binomial normal approximation of both runs).
  Reported alongside, not part of the rule: the correlation of per-problem accuracy between the two runs.

    python -m wsa.compare --authors <dir with exp8_math/eval> --ours <runs dir> [--limit N]
"""
import argparse
import glob
import json
import os

import numpy as np

ARMS = {"base": "unedited", "edit_authors": "p50c"}
SETS = {"math500": 4, "gsm8k": 2}


def authors_per_problem(root, run, name):
    per, cut, tot = {}, 0, 0
    for d in glob.glob(os.path.join(root, "exp8_math/eval", run, "parts", f"{name}_s*")):
        for r in json.load(open(os.path.join(d, f"{name}_per_problem.json"))):
            per[r["id"]] = r["correct"]
        s = json.load(open(os.path.join(d, "scores.json")))[name]
        cut += s["cut"]
        tot += s["total"]
    return per, cut, tot


def boot(diffs, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    m = diffs[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(a):
    report, mismatch = {}, False
    for arm, run in ARMS.items():
        for name, ns in SETS.items():
            ours_path = os.path.join(a.ours, arm, f"think35_{name}.per_problem.json")
            if not os.path.exists(ours_path):
                continue
            ours = {r["id"]: r["correct"] for r in json.load(open(ours_path)) if r["correct"]}
            theirs, cut_t, tot_t = authors_per_problem(a.authors, run, name)
            ids = [i for i in ours if i in theirs]
            o = np.array([np.mean(ours[i]) for i in ids])
            t = np.array([np.mean(theirs[i]) for i in ids])
            d = o - t
            lo, hi = boot(d)
            acc_bad = (lo > 0 or hi < 0) and abs(d.mean()) > 0.02
            sc = json.load(open(ours_path.replace(".per_problem.json", ".scores.json")))
            co, no = sc["cut"], sc["total"]
            pc, pt = co / no, cut_t / tot_t
            se = np.sqrt(pc * (1 - pc) / no + pt * (1 - pt) / tot_t)
            cut_bad = abs(pc - pt) > 1.96 * se and abs(pc - pt) > 0.02
            report[f"{arm}/{name}"] = {
                "n_problems": len(ids), "ours_acc": float(o.mean()), "authors_acc_same_problems": float(t.mean()),
                "diff": float(d.mean()), "diff_ci95": [lo, hi], "per_problem_corr": float(np.corrcoef(o, t)[0, 1]),
                "ours_cut": pc, "authors_cut_all_problems": pt, "acc_mismatch": bool(acc_bad),
                "cut_mismatch": bool(cut_bad)}
            mismatch |= acc_bad or cut_bad
    report["MISMATCH"] = bool(mismatch)
    print(json.dumps(report, indent=1))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--out")
    main(ap.parse_args())
