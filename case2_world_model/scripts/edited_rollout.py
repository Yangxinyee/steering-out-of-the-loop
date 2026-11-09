"""The held-out sustained dose-edit rollout (the companion code's rollout, unchanged) on a model whose state is edited.

    python scripts/edited_rollout.py <e03|n03> <run> <edit.npz|none> <method_tag> <seed>
        -> work/runs/table13/<world>/shifts_<method_tag>_s<seed>.parquet

Seed = the rollout seed of table13 (seed k for model sk), so the unedited arm reproduces the existing rollouts and every arm
shares the same noise. GPU. Synthetic data only; the companion code's work root is this project's work/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
COHORT = {"e03": "synth_e03", "n03": "synth_n03"}
key, run, edit_path, method, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
backend.configure(work_root=str(ROOT / "work"))
backend.configure(cohort=COHORT[key])                       # before the evaluation module is imported
if "-pnp-" in run or "-pnp" in run:
    backend.setdefault(mhat_file="atoms.mhat_prev.npy")
sys.path.insert(0, str(ROOT / "scripts"))

from backend import rollout as er  # noqa: E402
import stateedit  # noqa: E402

assert er.oe.COHORT == COHORT[key] and str(er.load_paths().work_root) == str(ROOT / "work")
orig_load = er.op.load_model


def load_model(r):
    model, cfg = orig_load(r)
    if edit_path != "none":
        stateedit.apply(model, stateedit.load(edit_path), gate_code=stateedit.insulin_code(ROOT / "work" / "bundles" / COHORT[key]))
        print("state edit applied:", model._state_edit, flush=True)
    return model, cfg


er.op.load_model = load_model
er.main(run, method, seed)
