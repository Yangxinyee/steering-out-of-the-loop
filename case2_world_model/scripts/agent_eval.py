"""Held-out evaluation of an agent run's submission : the edited plain model on the 600 held-out
stays the agent never saw.

    python scripts/agent_eval.py <agent run dir> [<method tag>]
1. the readout: the companion code's rollout on held_out with the submitted edits (the same rollout as the controls'
   and the agent's dose_edit tool), summarized against the world's truth -> results/agent_eval.jsonl
2. the factual fit: value NLL per value, teacher-forced on held_out, original and edited with the same random numbers
GPU. Synthetic data only.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
run_dir = Path(sys.argv[1]).resolve()
st = json.loads((run_dir / "state.json").read_text())
assert st.get("done"), f"{run_dir} has no final answer yet"
key, arm, run = st["world"], st["arm"], st["model_run"]
method = sys.argv[2] if len(sys.argv) > 2 else f"agent_{run_dir.name}"
COHORT = {"e03": "synth_e03", "n03": "synth_n03"}
WORLD = {"e03": "s+1_b1_e0.3", "n03": "s+0_b1_e0.3"}
backend.configure(work_root=str(ROOT / "work"))
backend.configure(cohort=COHORT[key])
sys.path.insert(0, str(ROOT / "agent"))

import polars as pl  # noqa: E402
import torch  # noqa: E402

from backend import baselines as cb  # noqa: E402
from backend import rollout as er  # noqa: E402
import fwd_job  # noqa: E402

sub = run_dir / "submission"
orig_load = er.op.load_model
touched: list[str] = []


def load_model(r):
    model, cfg = orig_load(r)
    sd, names = fwd_job.edited_state({k: v.detach().cpu() for k, v in model.state_dict().items()}, sub, arm)
    model.load_state_dict(sd)
    touched[:] = names
    print("edits applied:", names, flush=True)
    return model, cfg


er.op.load_model = load_model
er.main(run, method, 0)
r = cb.summarize(WORLD[key], method, 0, ROOT / "work" / "runs" / "table13" / WORLD[key])
rec = {"run": run_dir.name, "agent": st["agent"], "world": key, "arm": arm, "model_run": run, "method": method,
       "edited": touched, "edit_hash": st.get("edit_hash"), "passes": st["passes"], "python_runs": st["runs"]}
for e in ("insulin_x2", "insulin_half", "ne_x2", "ne_half"):
    x = r["edits"][e]
    rec[e] = {"mean_shift": x["mean_shift"], "true_shift": x.get("true_shift"), "ratio": x.get("ratio_model_to_true"),
              "n_stay_hours": x["n_stay_hours"]}

# the factual fit on held_out, both models with the same random numbers (fwd_job.factual on all held-out stays)
cuts = pl.read_parquet(er.oe.out_dir() / "cuts_held_out.parquet")
stays = sorted(cuts["subject_id"].to_list())
models, _ = fwd_job.load_models(run, ["original", "edited"], sub, arm)
with tempfile.TemporaryDirectory() as td:
    f = fwd_job.factual(models, "held_out", stays, Path(td))
rec["value_nll_held_out"] = {m: f[m]["value_nll_per_value"] for m in f}
rec["value_nll_change"] = round(f["edited"]["value_nll_per_value"] - f["original"]["value_nll_per_value"], 5)
rec["gpu_peak_mib"] = round(torch.cuda.max_memory_reserved() / 2**20)
out = Path(os.environ.get("AWR_EVAL_OUT", ROOT / "results" / "agent_eval.jsonl"))        # dry runs write elsewhere
with out.open("a") as fh:
    fh.write(json.dumps(rec) + "\n")
print(json.dumps(rec, indent=1))
