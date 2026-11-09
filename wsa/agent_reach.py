"""The frontier agents' edits on the reach and matched-completion tests (CPU, existing outputs).

For each agent's edit, on MATH-500 and GSM8K: the coverage of its correct answers on always-looping problems by the
unedited model under the protocol and its eight fixes (as wsa.coverage, per answer, 95% bootstrap over problems), and
the error of thinking off at the full budget minus the edit's at the lower completion rate of the two, under the one
selector of wsa.riskcov (1,000 bootstrap replicates over problems).

    python -m wsa.agent_reach        (writes results/agent_reach.json)
"""
import json
from pathlib import Path

import numpy as np

from wsa.coverage import SETS, boot
from wsa.rescue import load
from wsa.riskcov import curve, per_problem

RUNS = Path("results/runs")
AGENTS = {"gpt-6.1-sol": "agent_eval/openai_1/think35_{}", "Claude Opus 5.5": "agent_eval/claude_1/think35_{}",
          "published edit": "edit_authors/think35_{}"}
B = 1000


def main():
    out = {}
    for ds in ("MATH-500", "GSM8K"):
        kind, tag, decs, _ = SETS[ds]
        D = {k: load(RUNS, v.format(tag), kind) for k, v in decs.items()}
        base = D["base"]
        probs = sorted({p for p, _ in base})
        always = [p for p in probs if all(not base[(p, k)][1] for k in range(4))]
        cov = {p: any(D[d][(p, k)][0] and D[d][(p, k)][1] for d in D for k in range(4) if (p, k) in D[d]) for p in always}
        off = per_problem(D["nothink32k"])
        R = {}
        for nm, stem in AGENTS.items():
            E = load(RUNS, stem.format(tag), kind)
            right = {p: sum(1 for k in range(4) if (p, k) in E and E[(p, k)][0] and E[(p, k)][1]) for p in always}
            c = boot([right[p] * cov[p] for p in always], [right[p] for p in always])
            P = per_problem(E)
            ps = sorted(set(P) & set(off))
            fe, fo = curve(P, ps, [0.5])[1], curve(off, ps, [0.5])[1]
            cstar = min(fe, fo)

            def gap(sample):
                eo, _ = curve(off, sample, [cstar])
                ee, _ = curve(P, sample, [cstar])
                return None if eo[0] is None or ee[0] is None else eo[0] - ee[0]
            g0 = gap(ps)
            rng = np.random.default_rng(0)
            bs = [g for g in (gap(list(rng.choice(ps, len(ps)))) for _ in range(B)) if g is not None]
            R[nm] = {"correct_answers_always_looping": int(sum(right.values())), "coverage": list(c),
                     "completion": fe, "common_completion": cstar,
                     "thinking_off_minus_edit_error": [g0, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
            print(ds, f"{nm:16s} coverage {c[0]*100:.1f} [{c[1]*100:.1f}, {c[2]*100:.1f}]  completion {fe*100:.1f}  "
                  f"off - edit error at {cstar*100:.1f}%: {g0*100:+.1f} [{np.percentile(bs, 2.5)*100:.1f}, {np.percentile(bs, 97.5)*100:.1f}]")
        out[ds] = R
    Path("results/agent_reach.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
