"""The two frontier agents' edits on the test sets (runs/agent_eval/, author protocol think35, 4 samples): accuracy and
cut share, paired gains against the base model, the published agent's edit and the strongest decoding fix, and the
same released-answer measures as the rest of the paper (rescue: same-problem pairing; abstain: net accuracy of the
extra finished answers beyond the base model). CPU only, from the existing outputs.

    python -m wsa.agent_eval_report        (writes results/agent_eval.json)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wsa.abstain import boot, counts
from wsa.rescue import SETS, boot_ci, load

RUNS = Path("results/runs")
AGENTS = {"claude_1": "Claude Opus 5.5", "openai_1": "gpt-6.1-sol"}
TEST = {"MATH-500": "math500", "GSM8K": "gsm8k", "MedQA": "medqa"}
REFS = {"MATH-500": {"edit": "edit_authors/think35_math500", "nothink32k": "e3/test/nothink35_32k_math500",
                     "bf8k": "base/think35_bf8192_math500"},
        "GSM8K": {"edit": "edit_authors/think35_gsm8k", "nothink32k": "e3/test/nothink35_32k_gsm8k",
                  "bf8k": "base/think35_bf8192_gsm8k"},
        "MedQA": {"edit": "med_edit/think35_medqa", "t1": "med_base/card_think_medqa",
                  "bf8k": "med_base/think35_bf8192_medqa"}}


def per_problem(stem):
    return {r["id"]: np.mean(r["correct"]) for r in json.load(open(RUNS / f"{stem}.per_problem.json"))}


def paired(a, b):
    ids = sorted(set(a) & set(b))
    m, lo, hi = boot_ci([a[i] - b[i] for i in ids])
    return {"diff": m, "lo": lo, "hi": hi, "problems": len(ids)}


def main():
    out = {}
    for name, short in TEST.items():
        kind, base_stem, _ = SETS[name]
        base = load(RUNS, base_stem, kind)
        probs = sorted({p for p, _ in base})
        Cb = counts(base, probs)
        refs = {"base": base_stem, **REFS[name]}
        pp = {r: per_problem(s) for r, s in refs.items()}
        S = {}
        for ag in AGENTS:
            stem = f"agent_eval/{ag}/think35_{short}"
            sc = json.load(open(RUNS / f"{stem}.scores.json"))
            a = per_problem(stem)
            fx = load(RUNS, stem, kind)
            keys = [k for k in base if k in fx]
            resc = [k for k in keys if not base[k][1] and fx[k][1]]
            per_p = defaultdict(lambda: ([], []))
            for k in resc:
                per_p[k[0]][0].append(fx[k][0])
            for k in keys:
                if base[k][1]:
                    per_p[k[0]][1].append(base[k][0])
            pairs = [(np.mean(r), np.mean(b)) for r, b in per_p.values() if r and b]
            within = boot_ci([x - y for x, y in pairs])
            X = counts(fx, probs)

            def marg(i=None):
                x, y = (X, Cb) if i is None else (X[i], Cb[i])
                df = x[:, 0].sum() - y[:, 0].sum()
                return (x[:, 1].sum() - y[:, 1].sum()) / df if df > 0 else np.nan
            S[ag] = {"acc": sc["acc"], "cut_share": sc["cut_share"], "mean_tokens": sc["mean_tokens"],
                     "missing": sc["missing"], "no_answer_finished": sc["no_answer_finished"],
                     "vs": {r: paired(a, pp[r]) for r in refs},
                     "released": len(resc), "released_accuracy": float(np.mean([fx[k][0] for k in resc])) if resc else None,
                     "within_problem": {"problems": len(pairs),
                                        "released_acc": float(np.mean([x for x, _ in pairs])) if pairs else None,
                                        "own_acc": float(np.mean([y for _, y in pairs])) if pairs else None,
                                        "diff": within[0], "lo": within[1], "hi": within[2]},
                     "net_beyond_base": {"value": float(marg()), "ci": boot(marg, len(probs))}}
            v = S[ag]
            print(name, ag, f"acc {100 * v['acc']:.1f} cut {100 * v['cut_share']:.1f}",
                  {r: f"{100 * d['diff']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]" for r, d in v["vs"].items()},
                  f"released {v['released']} at {100 * (v['released_accuracy'] or 0):.1f}",
                  f"pair {v['within_problem']['problems']}: {100 * (v['within_problem']['released_acc'] or 0):.1f} vs "
                  f"{100 * (v['within_problem']['own_acc'] or 0):.1f} ({100 * (v['within_problem']['diff'] or 0):+.1f} "
                  f"[{100 * (v['within_problem']['lo'] or 0):+.1f}, {100 * (v['within_problem']['hi'] or 0):+.1f}])",
                  f"net {100 * v['net_beyond_base']['value']:.1f}")
        out[name] = S
    Path("results/agent_eval.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
