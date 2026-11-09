"""Risk-coverage curves under one shared selector (CPU, existing outputs).

Every setting's finished answers are scored the same way: the share of the setting's four samples of the same problem
that give the same answer (cut samples give none). Answers are taken in decreasing score; ties are broken at random, and
we report the expectation, which interpolates linearly within a tie block. Coverage is the share of all samples
answered; error is the strict error among them. Each fix G is compared with the agent's edit at coverages both reach,
on a 1-point grid up to the lower completion rate, and at that rate; 95% intervals by bootstrap over problems.

    python -m wsa.riskcov        (writes results/riskcov.json)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wsa.rescue import load

RUNS = Path("results/runs")
ARMS = {"base": "base/think35_{}", "t1": "base/card_think_{}", "bf8k": "base/think35_bf8192_{}",
        "loop": "base/think35_loop_{}", "eot_bias": "e3/test/think35_tb8_{}", "rep_penalty": "base/think35_rep11_{}",
        "prompt": "prompt/think35_pr2_{}", "meandiff": "meandiff_top64_a6/think35_{}", "nothink8k": "base/nothink35_{}",
        "nothink32k": "e3/test/nothink35_32k_{}", "authors_edit": "edit_authors/think35_{}"}
B = 1000


def per_problem(arm):
    """{problem: [(score, correct) for each finished sample], n samples}"""
    by_p = defaultdict(list)
    for (p, k), (c, f, a) in arm.items():
        by_p[p].append((c, f, a))
    out = {}
    for p, v in by_p.items():
        votes = Counter(a for _, f, a in v if f)
        out[p] = ([(votes[a] / len(v), bool(c)) for c, f, a in v if f], len(v))
    return out


def curve(P, probs, grid):
    """expected strict error at each coverage in grid (None beyond the setting's completion), and completion"""
    sc = np.array([s for p in probs for s, _ in P[p][0]])
    ok = np.array([c for p in probs for _, c in P[p][0]], float)
    N = sum(P[p][1] for p in probs)
    if len(sc) == 0:
        return [None] * len(grid), 0.0
    order = np.argsort(-sc, kind="stable")
    sc, ok = sc[order], ok[order]
    levels = np.unique(sc)[::-1]
    blk_n = np.array([(sc == l).sum() for l in levels])
    blk_c = np.array([ok[sc == l].sum() for l in levels])
    cum_n, cum_c = np.cumsum(blk_n), np.cumsum(blk_c)
    res = []
    for c in grid:
        n = c * N
        if n > cum_n[-1] + 1e-9 or n <= 0:
            res.append(None)
            continue
        i = int(np.searchsorted(cum_n, n - 1e-9))
        prev_n = cum_n[i - 1] if i else 0
        prev_c = cum_c[i - 1] if i else 0
        right = prev_c + (n - prev_n) * blk_c[i] / blk_n[i]
        res.append(1 - right / n)
    return res, cum_n[-1] / N


def main():
    out = {}
    for ds, tag in (("MATH-500", "math500"), ("GSM8K", "gsm8k")):
        kind = "math"
        P = {a: per_problem(load(RUNS, s.format(tag), kind)) for a, s in ARMS.items()}
        probs = sorted(set.intersection(*(set(v) for v in P.values())))
        grid = np.round(np.arange(0.01, 1.0001, 0.01), 2)
        R = {"problems": len(probs), "curves": {}, "vs_edit": {}}
        for a in ARMS:
            e, fin = curve(P[a], probs, grid)
            R["curves"][a] = {"completion": fin, "error": {f"{g:.2f}": v for g, v in zip(grid, e) if v is not None}}
        rng = np.random.default_rng(0)
        boots = [rng.choice(probs, len(probs), replace=True) for _ in range(B)]
        fe = R["curves"]["authors_edit"]["completion"]
        for a in ARMS:
            if a == "authors_edit":
                continue
            cstar = min(fe, R["curves"][a]["completion"])
            g2 = [g for g in grid if g <= cstar + 1e-9] + [cstar]
            def stat(ps):
                ea, _ = curve(P["authors_edit"], ps, g2)
                eg, _ = curve(P[a], ps, g2)
                d = [x - y for x, y in zip(eg, ea) if x is not None and y is not None]
                last = (eg[-1] - ea[-1]) if eg[-1] is not None and ea[-1] is not None else None
                return (float(np.mean(d)) if d else None), last
            m0, l0 = stat(probs)
            bs = [stat(list(ps)) for ps in boots]
            ms = [m for m, _ in bs if m is not None]
            ls = [l for _, l in bs if l is not None]
            R["vs_edit"][a] = {"common_coverage": cstar,
                               "mean_error_gap_over_common_range": [m0, float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))],
                               "error_gap_at_common_coverage": [l0, float(np.percentile(ls, 2.5)), float(np.percentile(ls, 97.5))]}
            v = R["vs_edit"][a]
            print(ds, f"{a:12s} c*={cstar*100:5.1f}  G-edit error gap: mean {m0*100:+.1f} [{v['mean_error_gap_over_common_range'][1]*100:+.1f}, {v['mean_error_gap_over_common_range'][2]*100:+.1f}]"
                  f"  at c* {l0*100:+.1f} [{v['error_gap_at_common_coverage'][1]*100:+.1f}, {v['error_gap_at_common_coverage'][2]*100:+.1f}]")
        out[ds] = R
    Path("results/riskcov.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
