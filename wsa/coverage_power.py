"""How sensitive is coverage? A spike-in power analysis from existing outputs (CPU).

A synthetic target keeps every correct answer of the agent's edit and adds m correct answers on problems no ruler
sample solves (beyond reach), placed at random (at most four answers per problem, the samples a model has), so that a
share f = m / (A + m) of its correct answers lies beyond reach. Sets: the always-looping problems of MATH-500 and GSM8K
(coverage: the share of correct answers whose problem some ruler sample solves, as wsa.coverage_null), and MMLU-Pro
math's never-finished questions (chance-corrected coverage at k = 16, as wsa.headroom_reach's leave-one-fix-out).
Two comparisons, each with a paired bootstrap over problems (1,000 replicates) and 40 random placements per f; a
placement is detected when the 95% interval excludes zero, and the minimum detectable share is the smallest f
detected in at least 80% of placements:
- against the edit itself (all fixes as the ruler): the floor, a target that differs from the edit only by the added
  answers;
- against release: each decoding fix in turn is left out, its own correct answers are scored against the other fixes,
  and the spiked edit, scored against the same ruler, must fall below them. A model is flagged as adding answers beyond
  reach when it falls below every fix, so the binding fix is the one whose own answers are covered least.

    python -m wsa.coverage_power        (writes results/coverage_power.json)
"""
import json
from pathlib import Path

import numpy as np

from wsa.coverage import SETS
from wsa.headroom_reach import FIXES, guessk, n_options, passk
from wsa.rescue import load

RUNS = Path("results/runs")
FRACS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3)
PLACEMENTS, BOOT = 40, 1000


def data(name):
    """per problem: the edit's correct answers, each fix's correct and finished counts (four samples), options"""
    right = lambda X, p: sum(1 for k in range(4) if X.get((p, k), (0, 0))[0] and X.get((p, k), (0, 0))[1])  # noqa: E731
    fin = lambda X, p: sum(1 for k in range(4) if X.get((p, k), (0, 0))[1])                                # noqa: E731
    if name != "MMLU-Pro math":
        kind, tag, decs, edit_stem = SETS[name]
        D = {k: load(RUNS, v.format(tag), kind) for k, v in decs.items()}
        base = D.pop("base")
        E = load(RUNS, edit_stem.format(tag), kind)
        probs = sorted({p for p, _ in base})
        sel = [p for p in probs if all(not base[(p, k)][1] for k in range(4))]
        opts = None
    else:
        opt = {x["id"]: n_options(x["prompt"]) for x in map(json.loads, open("results/data/mmlupro_math.jsonl"))}
        base = load(RUNS, "headroom/base/think35_mmlupro_math", "mc")
        E = load(RUNS, "headroom/edit/think35_mmlupro_math", "mc")
        D = {nm: load(RUNS, f"headroom/fix/{st}_mmlupro_math", "mc") for nm, st in FIXES.items()}
        sel = [p for p in sorted({p for p, _ in base}) if fin(base, p) == 0]
        opts = np.array([opt[p] for p in sel])
    r = np.array([right(E, p) for p in sel], float)
    C = {d: np.array([right(X, p) for p in sel], float) for d, X in D.items()}
    F = {d: np.array([fin(X, p) for p in sel], float) for d, X in D.items()}
    return r, C, F, opts


class Ruler:
    """coverage of answer weights w on bootstrap replicates ix, against the fixes in `fixes`"""
    def __init__(self, C, F, fixes, opts, ix):
        self.ix = ix
        c = sum(C[d] for d in fixes)
        self.c = c
        if opts is None:                                   # some ruler sample solves the problem
            self.P, self.G = (c > 0).astype(float), None
        else:                                              # chance-corrected coverage at k = all ruler samples
            n = 4 * len(fixes)
            f = sum(F[d] for d in fixes)
            self.P = np.array([passk(n, int(x), n) for x in c])
            self.G = np.array([guessk(n, int(fi), int(m), n) for fi, m in zip(f, opts)])

    def score(self, w):
        ix = self.ix
        cov = (w[ix] * self.P[ix]).sum(1) / w[ix].sum(1)
        if self.G is None:
            return cov
        g = (w[ix] * self.G[ix]).sum(1) / w[ix].sum(1)
        return (cov - g) / (1 - g)


def sweep(r, ruler, ref_scores, rng):
    """spike beyond-reach answers into the edit; detected when ref - spiked edit has a 95% interval above zero"""
    n, A = len(r), r.sum()
    cap = np.where(ruler.c == 0, 4 - r, 0).astype(int)
    slots = np.repeat(np.arange(n), cap)
    rows = []
    for fr in FRACS:
        m = int(round(fr / (1 - fr) * A))
        if m > len(slots):
            rows.append({"share": fr, "added": m, "feasible": False})
            continue
        det, diffs = 0, []
        for _ in range(PLACEMENTS):
            add = np.bincount(rng.choice(slots, m, replace=False), minlength=n)
            d = ref_scores - ruler.score(r + add)
            det += np.percentile(d, 2.5) > 0
            diffs.append(float(np.mean(d)))
        rows.append({"share": fr, "added": m, "feasible": True, "power": det / PLACEMENTS,
                     "mean_difference": float(np.mean(diffs))})
    mde = next((x["share"] for x in rows if x.get("feasible") and x["power"] >= 0.8), None)
    return mde, rows


def main():
    out = {}
    for name in ("MATH-500", "GSM8K", "MMLU-Pro math"):
        r, C, F, opts = data(name)
        rng = np.random.default_rng(0)
        ix = rng.integers(0, len(r), (BOOT, len(r)))
        allf = Ruler(C, F, list(C), opts, ix)
        R = {"problems": len(r), "edit_correct_answers": int(r.sum()), "edit_beyond_reach_answers": int(r[allf.c == 0].sum()),
             "edit_coverage": float(allf.score(r).mean()), "against_edit": {}, "against_release": {}}
        R["against_edit"]["minimum_detectable_share"], R["against_edit"]["rows"] = sweep(r, allf, allf.score(r), rng)
        for d in C:
            rest = [x for x in C if x != d]
            ru = Ruler(C, F, rest, opts, ix)
            own = ru.score(C[d])
            mde, rows = sweep(r, ru, own, rng)
            R["against_release"][d] = {"fix_own_coverage": float(own.mean()), "edit_coverage": float(ru.score(r).mean()),
                                       "minimum_detectable_share": mde, "rows": rows}
        rel = R["against_release"]
        R["binding_fix"] = min(rel, key=lambda d: rel[d]["fix_own_coverage"])
        mdes = [v["minimum_detectable_share"] for v in rel.values()]
        R["below_every_fix_minimum_share"] = None if None in mdes else max(mdes)
        out[name] = R
        print(name, "edit", round(R["edit_coverage"] * 100, 1), "| vs edit:", R["against_edit"]["minimum_detectable_share"],
              "| binding:", R["binding_fix"], "| below every fix at:", R["below_every_fix_minimum_share"])
        for d, v in rel.items():
            print(f"   {d:12s} own {v['fix_own_coverage']*100:5.1f} edit {v['edit_coverage']*100:5.1f} mde {v['minimum_detectable_share']}")
    Path("results/coverage_power.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
