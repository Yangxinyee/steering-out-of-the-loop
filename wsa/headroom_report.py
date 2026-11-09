"""MMLU-Pro results (the edit, and decoding-fix rows on MMLU-Pro math); CPU: the released-against-own test on MMLU-Pro math, a mathematics set with
headroom, and the medical-direction comparison on MMLU-Pro health with eight samples per question. Same definitions,
pairing and bootstrap as wsa.rescue (released = cut under the base model and finished under the edit, same question
and sample; own = the base model's finished answers on the same questions).

    python -m wsa.headroom_report [--runs results/runs]        (writes results/headroom.json)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from wsa.rescue import boot_ci, load

PAIRS = {"MMLU-Pro math, agent's edit": ("headroom/base/think35_mmlupro_math", "headroom/edit/think35_mmlupro_math"),
         "MMLU-Pro health, medical-direction edit, 8 samples": ("headroom/power_base/think35_mmlupro_health",
                                                                 "headroom/power_md/think35_mmlupro_health"),
         # decoding-fix reference rows on MMLU-Pro math, against the same base run
         **{f"MMLU-Pro math, {name}": ("headroom/base/think35_mmlupro_math", f"headroom/fix/{stem}_mmlupro_math")
            for name, stem in (("loop stop", "think35_loop"), ("budget forcing", "think35_bf8192"),
                               ("thinking off, 8k", "nothink35"), ("thinking off, full budget", "nothink35_32k"),
                               ("T = 1.0", "card_think"))}}


def compare(runs, bstem, fstem):
    B, F = load(runs, bstem, "mc"), load(runs, fstem, "mc")
    keys = [k for k in B if k in F]
    resc = [k for k in keys if not B[k][1] and F[k][1]]
    per_p = defaultdict(lambda: ([], []))
    for k in resc:
        per_p[k[0]][0].append(F[k][0])
    for k in keys:
        if B[k][1]:
            per_p[k[0]][1].append(B[k][0])
    pairs = [(np.mean(r), np.mean(b)) for r, b in per_p.values() if r and b]
    d, lo, hi = boot_ci([x - y for x, y in pairs])
    acc = lambda D: float(np.mean([v[0] for v in D.values()]))              # noqa: E731  authors' reading
    strict = lambda D: float(np.mean([v[0] and v[1] for v in D.values()]))  # noqa: E731
    fin = lambda D: float(np.mean([v[1] for v in D.values()]))              # noqa: E731
    own = [B[k][0] for k in keys if B[k][1]]
    probs = {p for p, _ in keys}
    return {"questions": len(probs), "samples": len(keys) // max(len(probs), 1),
            "base": {"acc": acc(B), "strict": strict(B), "finished": fin(B)},
            "fix": {"acc": acc(F), "strict": strict(F), "finished": fin(F)},
            "own_finished_accuracy": float(np.mean(own)) if own else None, "released": len(resc),
            "released_accuracy": float(np.mean([F[k][0] for k in resc])) if resc else None,
            "within_problem": {"problems": len(pairs), "released_acc": float(np.mean([x for x, _ in pairs])) if pairs else None,
                               "own_acc": float(np.mean([y for _, y in pairs])) if pairs else None,
                               "diff": d, "lo": lo, "hi": hi}}


def main(a):
    runs = Path(a.runs)
    out = {}
    for name, (b, f) in (PAIRS if not a.test else {"test: MMLU-Pro health, 4 samples": (
            "mmlu_base/think35_mmlupro_health", "mmlu_md_a6/think35_mmlupro_health")}).items():
        if not (runs / f"{f}.per_problem.json").exists():
            print("missing", f)
            continue
        R = compare(runs, b, f)
        out[name] = R
        w = R["within_problem"]
        print(f"{name}: {R['questions']} questions x {R['samples']}; base {R['base']['acc']*100:.1f} (finished "
              f"{R['base']['finished']*100:.1f}), fix {R['fix']['acc']*100:.1f}; own finished {R['own_finished_accuracy']*100:.1f}; "
              f"released {R['released']} right {R['released_accuracy']*100:.1f}; same problems ({w['problems']}): "
              f"{w['released_acc']*100:.1f} vs {w['own_acc']*100:.1f}, {w['diff']*100:+.1f} [{w['lo']*100:.1f}, {w['hi']*100:.1f}]")
    if not a.test:
        Path("results/headroom.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--test", action="store_true")
    main(ap.parse_args())
