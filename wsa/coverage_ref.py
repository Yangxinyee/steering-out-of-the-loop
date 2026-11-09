"""A reference scale for the coverage of the agent's edit (CPU, existing outputs).

On the always-looping problems of Qwen3.5-0.8B (all four samples under the authors' protocol cut), the unedited 0.8B's
samples under the eight fixes (32 per problem; the protocol itself finishes none of these problems) are asked to cover
the correct answers of two targets: the agent's edit, and Qwen3.5-2B under the same protocol, a larger model that
solves problems the 0.8B cannot. A target's correct answer (finished and right, four samples per problem) is covered
when some 0.8B sample finishes the same problem correctly. Coverage against k uses the unbiased pass@k estimator over
the 32 samples, 1 - C(32 - c, k) / C(32, k), with c the problem's correct 0.8B samples. The 2B reference is also
matched to the edit's solved count: its N most reliably solved problems (highest share of correct samples; ties broken
at random), N the edit's solved count, which favours high coverage. 95% intervals by bootstrap over problems.

    python -m wsa.coverage_ref        (writes results/coverage_ref.json)
"""
import json
from math import comb
from pathlib import Path

import numpy as np

from wsa.coverage import SETS
from wsa.rescue import load

RUNS = Path("results/runs")
TWO_B = {"MATH-500": "c2b/base/think35_math500", "GSM8K": "c2b/base/think35_gsm8k"}


def boot_ratio(num, den, n=10000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (n, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return [float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def main():
    out = {}
    for ds in ("MATH-500", "GSM8K"):
        kind, tag, decs, edit_stem = SETS[ds]
        D = {k: load(RUNS, v.format(tag), kind) for k, v in decs.items()}
        base = D.pop("base")
        probs = sorted({p for p, _ in base})
        always = [p for p in probs if all(not base[(p, k)][1] for k in range(4))]
        n_s = 4 * len(D)
        c = {p: sum(D[d][(p, k)][0] and D[d][(p, k)][1] for d in D for k in range(4) if (p, k) in D[d]) for p in always}
        targets = {"agent's edit": load(RUNS, edit_stem.format(tag), kind), "Qwen3.5-2B": load(RUNS, TWO_B[ds], kind)}
        R = {"always_looping": len(always), "samples_per_problem": n_s, "targets": {}}
        for name, T in targets.items():
            right = {p: sum(1 for k in range(4) if (p, k) in T and T[(p, k)][0] and T[(p, k)][1]) for p in always}
            solved = [p for p in always if right[p]]
            curve = {}
            for k in range(1, n_s + 1):
                pk = {p: 1 - comb(n_s - c[p], k) / comb(n_s, k) for p in always}
                curve[k] = {"per_answer": float(sum(right[p] * pk[p] for p in always) / sum(right.values())),
                            "per_problem": float(np.mean([pk[p] for p in solved]))}
            cov = {p: c[p] > 0 for p in always}
            R["targets"][name] = {"solved": len(solved), "correct_answers": int(sum(right.values())),
                                  "per_answer": boot_ratio([right[p] * cov[p] for p in always], [right[p] for p in always]),
                                  "per_problem": boot_ratio([cov[p] for p in solved], [1] * len(solved)),
                                  "curve": curve, "_right": right}
        # 2B matched to the edit's solved count: its most reliably solved problems
        n_edit = R["targets"]["agent's edit"]["solved"]
        r2 = R["targets"]["Qwen3.5-2B"]["_right"]
        rng = np.random.default_rng(0)
        order = sorted([p for p in always if r2[p]], key=lambda p: (-r2[p], rng.random()))[:n_edit]
        cov = {p: c[p] > 0 for p in always}
        keep = set(order)
        r2m = {p: (r2[p] if p in keep else 0) for p in always}
        curve_m = {}
        for k in range(1, n_s + 1):
            pk = {p: 1 - comb(n_s - c[p], k) / comb(n_s, k) for p in always}
            curve_m[k] = {"per_answer": float(sum(r2m[p] * pk[p] for p in always) / sum(r2m.values())),
                          "per_problem": float(np.mean([pk[p] for p in order]))}
        R["targets"]["Qwen3.5-2B, matched"] = {
            "solved": len(order), "correct_answers": int(sum(r2[p] for p in order)),
            "per_answer": boot_ratio([r2[p] * cov[p] for p in order], [r2[p] for p in order]),
            "per_problem": boot_ratio([cov[p] for p in order], [1] * len(order)), "curve": curve_m, "_right": r2m}
        # edit minus 2B coverage at matched k, bootstrap over always-looping problems (both targets resampled together)
        re_, r2_ = R["targets"]["agent's edit"]["_right"], R["targets"]["Qwen3.5-2B, matched"]["_right"]
        A = np.array([re_[p] for p in always], float)
        Bv = np.array([r2_[p] for p in always], float)
        C = np.array([c[p] for p in always])
        idx = np.random.default_rng(1).integers(0, len(always), (10000, len(always)))
        R["edit_minus_2b_matched"] = {}
        for k in (1, 2, 4, 8, 16, 32):
            pk = np.array([1 - comb(n_s - ci, k) / comb(n_s, k) for ci in C])
            def d(ix):
                return (A[ix] * pk[ix]).sum() / A[ix].sum() - (Bv[ix] * pk[ix]).sum() / Bv[ix].sum()
            bs = np.array([d(ix) for ix in idx])
            R["edit_minus_2b_matched"][k] = [float(d(np.arange(len(always)))), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        for t in R["targets"].values():
            t.pop("_right", None)
        out[ds] = R
        print(ds, f"always-looping {len(always)}, 0.8B samples per problem {n_s}")
        for name, t in R["targets"].items():
            a, q = t["per_answer"], t["per_problem"]
            extra = f"  k=4: {t['curve'][4]['per_answer']*100:.1f}  k=8: {t['curve'][8]['per_answer']*100:.1f}" if "curve" in t else ""
            print(f"  {name:22s} solved {t['solved']:4d}  answers {t['correct_answers']:5d}  per answer {a[0]*100:.1f} [{a[1]*100:.1f}, {a[2]*100:.1f}]"
                  f"  per problem {q[0]*100:.1f} [{q[1]*100:.1f}, {q[2]*100:.1f}]{extra}")
        print("  edit - matched 2B per answer by k:", {k: f"{v[0]*100:+.1f} [{v[1]*100:.1f}, {v[2]*100:.1f}]" for k, v in R["edit_minus_2b_matched"].items()})
    Path("results/coverage_ref.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
