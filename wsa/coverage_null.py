"""Leave-one-fix-out null for the coverage ruler (CPU, existing outputs).

A decoding fix keeps the 0.8B's weights, so its correct answers are released answers by construction. On the
always-looping problems, each fix (or the thinking-off family, both budgets) in turn is the target and the remaining
fixes are the ruler: the share of the target's correct answers (finished and right, four samples per problem) whose
problem some ruler sample also finishes correctly. The agent's edit and Qwen3.5-2B (all problems it solves, unmatched;
its random match to the edit's solved count gives the same value, wsa.coverage_match) are scored against the same
ruler, so all three sit on one scale. 95% intervals by bootstrap over always-looping problems; differences use the
same replicate for both targets.

    python -m wsa.coverage_null        (writes results/coverage_null.json)
"""
import json
from math import comb
from pathlib import Path

import numpy as np

from wsa.coverage import SETS
from wsa.coverage_ref import TWO_B
from wsa.rescue import load

RUNS = Path("results/runs")
N_BOOT = 10000


def main():
    out = {}
    for ds in ("MATH-500", "GSM8K"):
        kind, tag, decs, edit_stem = SETS[ds]
        D = {k: load(RUNS, v.format(tag), kind) for k, v in decs.items()}
        base = D.pop("base")
        probs = sorted({p for p, _ in base})
        always = [p for p in probs if all(not base[(p, k)][1] for k in range(4))]
        right = lambda X: np.array([sum(1 for k in range(4) if (p, k) in X and X[(p, k)][0] and X[(p, k)][1])
                                    for p in always], float)
        R_fix = {d: right(X) for d, X in D.items()}
        targets_fixed = {"agent's edit": right(load(RUNS, edit_stem.format(tag), kind)),
                         "Qwen3.5-2B": right(load(RUNS, TWO_B[ds], kind))}
        idx = np.random.default_rng(0).integers(0, len(always), (N_BOOT, len(always)))
        leave = {d: (d,) for d in D}
        leave["thinking off (both budgets)"] = ("nothink8k", "nothink32k")
        R = {"always_looping": len(always), "fixes": list(D), "left_out": {}}
        for name, out_set in leave.items():
            rest = [d for d in D if d not in out_set]
            c = sum(R_fix[d] for d in rest)                   # correct ruler samples per problem
            n_s = 4 * len(rest)
            cov = (c > 0).astype(float)
            pk4 = np.array([1 - comb(n_s - int(ci), 4) / comb(n_s, 4) for ci in c])
            tg = {"left-out fix": sum(R_fix[d] for d in out_set), **targets_fixed}
            row = {"ruler": rest, "ruler_samples": n_s, "targets": {}}
            for t, r in tg.items():
                if r.sum() == 0:
                    continue
                bs = (r[idx] * cov[idx]).sum(1) / np.maximum(r[idx].sum(1), 1e-12)
                row["targets"][t] = {"answers": int(r.sum()), "problems": int((r > 0).sum()),
                                     "coverage": [float((r * cov).sum() / r.sum()), float(np.percentile(bs, 2.5)),
                                                  float(np.percentile(bs, 97.5))],
                                     "coverage_k4": float((r * pk4).sum() / r.sum())}
            e, f, b = targets_fixed["agent's edit"], tg["left-out fix"], targets_fixed["Qwen3.5-2B"]
            share = lambda r: (r[idx] * cov[idx]).sum(1) / np.maximum(r[idx].sum(1), 1e-12)
            for key, (x, y, nx, ny) in {"edit_minus_fix": (e, f, "agent's edit", "left-out fix"),
                                        "fix_minus_2b": (f, b, "left-out fix", "Qwen3.5-2B")}.items():
                if x.sum() and y.sum():
                    d_bs = share(x) - share(y)
                    row[key] = [row["targets"][nx]["coverage"][0] - row["targets"][ny]["coverage"][0],
                                float(np.percentile(d_bs, 2.5)), float(np.percentile(d_bs, 97.5))]
            R["left_out"][name] = row
        out[ds] = R
        print(ds, f"always-looping {len(always)}")
        for name, row in R["left_out"].items():
            t = row["targets"]
            s = "  ".join(f"{k}: {v['coverage'][0]*100:.1f} [{v['coverage'][1]*100:.1f}, {v['coverage'][2]*100:.1f}]"
                          f" (n={v['answers']}, k4 {v['coverage_k4']*100:.1f})" for k, v in t.items())
            dd = row.get("edit_minus_fix")
            print(f"   leave out {name:28s} ruler {row['ruler_samples']:2d}  {s}"
                  + (f"  edit-fix {dd[0]*100:+.1f} [{dd[1]*100:.1f}, {dd[2]*100:.1f}]" if dd else ""))
    Path("results/coverage_null.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
