"""The audit's tests on a known install: the distilled 0.8B (wsa.sft; ckpt/sft_ref3), Qwen3.5-0.8B fine-tuned
on the 9B teacher's solutions, scored with the definitions used for the agent's edit (CPU, existing outputs).

- MedQA: accuracy against the base model, T = 1.0 and the agent's edit, paired by question (mean over the
  four samples, bootstrap over questions as wsa.rescue); released against own answers on the same questions.
- MMLU-Pro math: the same, against the base model, the edit and every decoding fix run there.
Released = cut under the base model and finished under the target, same question and sample (wsa.headroom_report).
The coverage tests are wsa.coverage_sft (MATH-500, GSM8K) and wsa.headroom_reach --target (MMLU-Pro math).

    python -m wsa.sft_battery [--stem sft_ref3/eval/think35_{}]        (writes results/sft_battery.json)
"""
import argparse
import json
from pathlib import Path

import numpy as np

from wsa.headroom_report import compare
from wsa.rescue import boot_ci

RUNS = Path("results/runs")
SETS = {"MedQA": ("medqa", "med_base/think35_medqa", {"T = 1.0": "med_base/card_think_medqa",
                                                      "agent's edit": "med_edit/think35_medqa",
                                                      "loop stop": "med_base/think35_loop_medqa",
                                                      "budget forcing": "med_base/think35_bf8192_medqa"}),
        "MMLU-Pro math": ("mmlupro_math", "headroom/base/think35_mmlupro_math",
                          {"agent's edit": "headroom/edit/think35_mmlupro_math",
                           "T = 1.0": "headroom/fix/card_think_mmlupro_math",
                           "loop stop": "headroom/fix/think35_loop_mmlupro_math",
                           "budget forcing": "headroom/fix/think35_bf8192_mmlupro_math",
                           "thinking off, 8k": "headroom/fix/nothink35_mmlupro_math",
                           "thinking off, full budget": "headroom/fix/nothink35_32k_mmlupro_math"})}


def per_q(stem):
    return {r["id"]: float(np.mean(r["correct"])) for r in json.load(open(RUNS / f"{stem}.per_problem.json"))}


def paired(a, b):
    A, B = per_q(a), per_q(b)
    return list(boot_ci([A[i] - B[i] for i in sorted(set(A) & set(B))]))


def main(a):
    out = {}
    for ds, (tag, base, others) in SETS.items():
        stem = a.stem.format(tag)
        if not (RUNS / f"{stem}.per_problem.json").exists():
            print(ds, "no run", stem)
            continue
        R = compare(RUNS, base, stem)
        R["reference_minus"] = {"base": paired(stem, base)}
        R["reference_minus"].update({nm: paired(stem, s) for nm, s in others.items() if (RUNS / f"{s}.per_problem.json").exists()})
        R["stem"] = stem
        out[ds] = R
        w = R["within_problem"]
        print(f"{ds}: reference {R['fix']['acc']*100:.1f} (finished {R['fix']['finished']*100:.1f}), base {R['base']['acc']*100:.1f}; "
              f"released {R['released']} right {R['released_accuracy']*100:.1f}, own finished {R['own_finished_accuracy']*100:.1f}; "
              f"same questions ({w['problems']}): {w['released_acc']*100:.1f} vs {w['own_acc']*100:.1f}, {w['diff']*100:+.1f} "
              f"[{w['lo']*100:.1f}, {w['hi']*100:.1f}]")
        for nm, (d, lo, hi) in R["reference_minus"].items():
            print(f"   reference - {nm}: {d*100:+.1f} [{lo*100:.1f}, {hi*100:.1f}]")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="sft_ref3/eval/think35_{}")
    ap.add_argument("--out", default="results/sft_battery.json")
    main(ap.parse_args())
