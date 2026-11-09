"""Readouts on interventions no agent ever queried: the companion code's held-out rollout with other dose factors,
other outcomes and other start hours, for any corrected model (an agent's submission, a state edit, a fine-tuned run).

    python scripts/generalize.py <e03|n03> <run> <loader> <tag> <seed> <design>
      loader: none | state:<edit.npz> | agent:<agent run dir>
      design: doses   -- start hour 12: insulin x0, x1.5, x3 on finger-stick glucose, insulin x2 on laboratory glucose,
                         norepinephrine x0, x1.5, x3 on arterial MAP, norepinephrine x2 on cuff MAP
              starts  -- the four original edits (x2, x1/2 of each drug) started at hour 6 and at hour 16
    -> work/runs/general/<world>/<tag>_s<seed>_<design>.parquet  (one row per stay, hour, edit; hours are labelled 13..24
       relative to the start, as in table13; column `start` is the start hour)

The agents' forward tool only ever ran the four edits from hour 12 on finger-stick glucose and arterial MAP; nothing here was
available to them. Paths, common random numbers and the `reached` rule are the rollout's. GPU. Synthetic data only.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
COHORT = {"e03": "synth_e03", "n03": "synth_n03"}
WORLD = {"e03": "s+1_b1_e0.3", "n03": "s+0_b1_e0.3"}
key, run, loader, tag, seed, design = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6]
backend.configure(work_root=str(ROOT / "work"))
backend.configure(cohort=COHORT[key])
if "-pnp" in run:
    backend.setdefault(mhat_file="atoms.mhat_prev.npy")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agent"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from backend import rollout as er  # noqa: E402

DOSES = {
    "insulin_x0": ("insulin", 0.0, "glucose_finger", "G"), "insulin_x1.5": ("insulin", 1.5, "glucose_finger", "G"),
    "insulin_x3": ("insulin", 3.0, "glucose_finger", "G"), "insulin_x2_lab": ("insulin", 2.0, "glucose_lab", "G"),
    "ne_x0": ("norepinephrine", 0.0, "map_arterial", "M"), "ne_x1.5": ("norepinephrine", 1.5, "map_arterial", "M"),
    "ne_x3": ("norepinephrine", 3.0, "map_arterial", "M"), "ne_x2_cuff": ("norepinephrine", 2.0, "map_cuff", "M")}
ORIGINAL = dict(er.EDITS)
STARTS = tuple(int(x) for x in os.environ.get("AWR_STARTS", "6,16").split(","))       # AWR_STARTS=6: one start hour
#: AWR_STAYS=lo:hi -- only these held-out stays (in cut order); every stay's random numbers are seeded by the stay, so the
#: pieces of a split run give exactly the rows of the whole run
STAYS = os.environ.get("AWR_STAYS")
SUFFIX = os.environ.get("AWR_SUFFIX", "")
#: AWR_ONLY=insulin_x0,ne_x0 -- only these edits of the design (the x0 diagnostic); AWR_ZERO_RATE=1 -- a stopped dose reaches
#: the expected-dose pathway as rate 0 (its log taken at 1e-9 units, so the pathway's dose-units input is minus the expected
#: dose) instead of an undefined log rate, which drops the atom from the pathway (the default, and every other evaluation)
ONLY = [e for e in os.environ.get("AWR_ONLY", "").split(",") if e]
ZERO_RATE = os.environ.get("AWR_ZERO_RATE") == "1"

orig_load = er.op.load_model


def load_model(r):
    model, cfg = orig_load(r)
    if loader.startswith("state:"):
        import stateedit
        stateedit.apply(model, stateedit.load(loader[6:]), gate_code=stateedit.insulin_code(ROOT / "work" / "bundles" / COHORT[key]))
        print("state edit applied:", model._state_edit, flush=True)
    elif loader.startswith("agent:"):
        import json

        import fwd_job
        d = Path(loader[6:]).resolve()
        arm = json.loads((d / "state.json").read_text())["arm"]
        sd, names = fwd_job.edited_state({k: v.detach().cpu() for k, v in model.state_dict().items()}, d / "submission", arm)
        model.load_state_dict(sd)
        print("agent edits applied:", len(names), "tensors", flush=True)
    return model, cfg


er.op.load_model = load_model
if ZERO_RATE:
    def _log_rate(raw):
        return np.log(np.maximum(np.asarray(raw, dtype=np.float64), 1e-9)).astype(np.float32)
    er.op.log_rate = _log_rate


def set_edits(edits: dict) -> None:
    er.EDITS = edits
    er.ARMS = ("factual", *edits)


def shifted(cuts: pl.DataFrame, root: Path, start: int) -> tuple[pl.DataFrame, dict]:
    """The cut moved from ICU hour 12 to `start`: the first bundle at or after admission + start hours (the rule that gives
    table13's cut at hour 12, checked below)."""
    from backend import BundleStore
    store = BundleStore(root, "held_out")
    bt = np.asarray(store.a["bundle_t"]).astype(np.int64)
    t_new = (cuts["t_cut"] + (start - 12) * 3600).to_list()
    cut_map = {int(lo): int(lo + np.searchsorted(bt[lo:hi], t, side="left"))
               for lo, hi, t in zip(cuts["bundle_lo"], cuts["bundle_hi"], t_new, strict=True)}
    return cuts.with_columns(pl.Series("t_cut", t_new)), cut_map


def main() -> None:
    model, cfg, root, ids, by_id, cut_map, spec, _ = er.op.setup(run)
    cuts = er.load(cut_map, spec)
    if STAYS:
        lo, hi = (int(x) for x in STAYS.split(":"))
        cuts = cuts.slice(lo, hi - lo)
    # the rule used to move the cut reproduces table13's cut at hour 12
    _, same = shifted(cuts, root, 12)
    assert all(same[int(lo)] == int(cut_map[int(lo)]) for lo in cuts["bundle_lo"]), "cut rule does not reproduce hour 12"
    out = ROOT / "work" / "runs" / "general" / WORLD[key]
    out.mkdir(parents=True, exist_ok=True)
    t0, frames = time.time(), []
    if design == "doses":
        set_edits({e: v for e, v in DOSES.items() if not ONLY or e in ONLY})
        frames.append(pl.DataFrame(er.main_rows(model, root, ids, cut_map, spec, cuts, seed)).with_columns(pl.lit(12).alias("start")))
    elif design == "starts":
        set_edits(ORIGINAL)
        for start in STARTS:
            c2, m2 = shifted(cuts, root, start)
            frames.append(pl.DataFrame(er.main_rows(model, root, ids, m2, spec, c2, seed)).with_columns(pl.lit(start).alias("start")))
            print("start", start, "done", round((time.time() - t0) / 60, 1), "min", flush=True)
    else:
        raise SystemExit(f"unknown design {design}")
    f = pl.concat(frames)
    path = out / f"{tag}_s{seed}_{design}{SUFFIX}.parquet"
    f.write_parquet(path)
    print("wrote", path, f.height, "rows;", round((time.time() - t0) / 60, 1), "min; gpu peak MiB",
          round(torch.cuda.max_memory_reserved() / 2**20), flush=True)


if __name__ == "__main__":
    main()
