"""Fairer matching of the Qwen3.5-2B coverage reference (CPU, existing outputs).

`wsa.coverage_ref` matched the 2B to the edit's solved count by keeping its most reliably solved always-looping
problems, which favours high coverage. Here the 2B keeps N of the problems it solves (N = the edit's solved count):
  random     - N drawn uniformly;
  difficulty - drawn within difficulty strata so that the strata hold as many problems as the edit solves in each
               (MATH-500: the dataset's level 1-5; GSM8K: lines in the reference solution, 2, 3, 4, 5, 6 or more;
               labels in results/data/difficulty.jsonl). A stratum where the 2B solves fewer problems than the edit
               gives all of them.
Coverage of a target's correct answers (finished and right, four samples per problem) against the unedited 0.8B's
32 samples under the eight fixes: per answer, with the unbiased pass@k estimator over the 32 samples (k = 32 is
"covered by any sample"). Point estimates average 4,000 draws of the subset; 95% intervals come from 4,000 bootstrap
replicates over always-looping problems, each with its own draw, so they include the draw. Differences (edit minus
reference) use the same replicate for both.

    python -m wsa.coverage_match        (writes results/coverage_match.json)
"""
import json
from math import comb
from pathlib import Path

import numpy as np

from wsa.coverage import SETS
from wsa.coverage_ref import TWO_B
from wsa.rescue import load

RUNS = Path("results/runs")
KS = (1, 2, 4, 8, 16, 32)
N_DRAW = N_BOOT = 4000


def stratum(ds, d):
    return d if ds == "MATH-500" else min(d, 6)


def main():
    diff = {}
    for line in open("results/data/difficulty.jsonl"):
        r = json.loads(line)
        diff[(r["set"], r["id"])] = r["difficulty"]
    out = {}
    for ds in ("MATH-500", "GSM8K"):
        kind, tag, decs, edit_stem = SETS[ds]
        D = {k: load(RUNS, v.format(tag), kind) for k, v in decs.items()}
        base = D.pop("base")
        probs = sorted({p for p, _ in base})
        always = [p for p in probs if all(not base[(p, k)][1] for k in range(4))]
        n_s = 4 * len(D)
        c = np.array([sum(D[d][(p, k)][0] and D[d][(p, k)][1] for d in D for k in range(4) if (p, k) in D[d])
                      for p in always])
        E, T = load(RUNS, edit_stem.format(tag), kind), load(RUNS, TWO_B[ds], kind)
        right = lambda X: np.array([sum(1 for k in range(4) if (p, k) in X and X[(p, k)][0] and X[(p, k)][1])
                                    for p in always], float)
        re_, r2 = right(E), right(T)
        st = np.array([stratum(ds, diff[(ds, p)]) for p in always])
        pk = {k: np.array([1 - comb(n_s - ci, k) / comb(n_s, k) for ci in c]) for k in KS}

        def cover(r, ix, w):                 # per-answer coverage of target r on rows ix with subset weights w
            return {k: float((r[ix] * w * pk[k][ix]).sum() / (r[ix] * w).sum()) for k in KS}

        def pick(ix, rng, how):              # 0/1 weights over rows ix: the 2B's matched subset
            w = np.zeros(len(ix))
            pos2 = np.flatnonzero(r2[ix] > 0)
            if how == "random":
                n = int((re_[ix] > 0).sum())
                w[rng.choice(pos2, min(n, len(pos2)), replace=False)] = 1
            else:
                for s in np.unique(st[ix]):
                    n = int(((re_[ix] > 0) & (st[ix] == s)).sum())
                    cand = pos2[st[ix][pos2] == s]
                    w[rng.choice(cand, min(n, len(cand)), replace=False)] = 1
            return w

        full = np.arange(len(always))
        R = {"always_looping": len(always), "samples_per_problem": n_s,
             "edit": {"solved": int((re_ > 0).sum()), "coverage": cover(re_, full, np.ones(len(full)))},
             "strata": {}, "references": {}}
        for s in np.unique(st):
            R["strata"][int(s)] = {"edit_solved": int(((re_ > 0) & (st == s)).sum()),
                                   "2b_solved": int(((r2 > 0) & (st == s)).sum()), "problems": int((st == s).sum())}
        for how in ("random", "difficulty"):
            rng = np.random.default_rng(0)
            draws = [pick(full, rng, how) for _ in range(N_DRAW)]
            point = {k: float(np.mean([cover(r2, full, w)[k] for w in draws])) for k in KS}
            rng = np.random.default_rng(1)
            bs_ref, bs_diff = {k: [] for k in KS}, {k: [] for k in KS}
            for _ in range(N_BOOT):
                ix = rng.integers(0, len(always), len(always))
                w = pick(ix, rng, how)
                ce, cr = cover(re_, ix, np.ones(len(ix))), cover(r2, ix, w)
                for k in KS:
                    bs_ref[k].append(cr[k])
                    bs_diff[k].append(ce[k] - cr[k])
            R["references"][f"Qwen3.5-2B, {how}"] = {
                "solved": int(np.mean([w.sum() for w in draws])),
                "coverage": {k: [point[k], float(np.percentile(bs_ref[k], 2.5)), float(np.percentile(bs_ref[k], 97.5))]
                             for k in KS},
                "edit_minus_ref": {k: [R["edit"]["coverage"][k] - point[k], float(np.percentile(bs_diff[k], 2.5)),
                                       float(np.percentile(bs_diff[k], 97.5))] for k in KS}}
        out[ds] = R
        print(ds, f"always-looping {len(always)}; edit solves {R['edit']['solved']}; edit coverage k=4 "
                  f"{R['edit']['coverage'][4]*100:.1f}, k=32 {R['edit']['coverage'][32]*100:.1f}")
        print("   strata (edit solved / 2B solved / problems):",
              {s: (v["edit_solved"], v["2b_solved"], v["problems"]) for s, v in R["strata"].items()})
        for name, v in R["references"].items():
            cv, dv = v["coverage"], v["edit_minus_ref"]
            print(f"   {name:26s} k=32 {cv[32][0]*100:.1f} [{cv[32][1]*100:.1f}, {cv[32][2]*100:.1f}]  k=4 {cv[4][0]*100:.1f}"
                  f"   edit - ref: k=32 {dv[32][0]*100:+.1f} [{dv[32][1]*100:.1f}, {dv[32][2]*100:.1f}]"
                  f"  k=4 {dv[4][0]*100:+.1f} [{dv[4][1]*100:.1f}, {dv[4][2]*100:.1f}]")
    Path("results/coverage_match.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
