"""Readout of held-out rollouts, as the companion code's evaluation summarizes them: per edit, the mean shift over
reached stay-hours, the true shift and the ratio model / truth with a patient bootstrap.

    python scripts/readout.py <e03|n03> <method_tag> <seed> [<dir>]     # default dir: work/runs/table13/<world>
Appends to results/readout.jsonl and prints the four edits. CPU.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
backend.configure(work_root=str(ROOT / "work"))
WORLD = {"e03": ("s+1_b1_e0.3", "synth_e03"), "n03": ("s+0_b1_e0.3", "synth_n03")}
key, method, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
backend.configure(cohort=WORLD[key][1])

from backend import baselines as cb  # noqa: E402

d = Path(sys.argv[4]) if len(sys.argv) > 4 else ROOT / "work" / "runs" / "table13" / WORLD[key][0]
r = cb.summarize(WORLD[key][0], method, seed, d)
rec = {"key": key, "method": method, "seed": seed, "dir": str(d.relative_to(ROOT)) if d.is_relative_to(ROOT) else str(d)}
for e in ("insulin_x2", "insulin_half", "ne_x2", "ne_half"):
    x = r["edits"][e] if "edits" in r else r[e]
    rec[e] = {"mean_shift": x["mean_shift"], "true_shift": x.get("true_shift"), "ratio": x.get("ratio_model_to_true"), "n_stay_hours": x["n_stay_hours"]}
with (ROOT / "results" / "readout.jsonl").open("a") as f:
    f.write(json.dumps(rec) + "\n")
print(json.dumps(rec))
