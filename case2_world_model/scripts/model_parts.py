"""Every loss part, and their total, for any corrected model, teacher-forced on the 600 held-out stays; compared with the
plain model of the same world. Extends scripts/agent_parts.py (agents only) to the controls.

    python scripts/model_parts.py <e03|n03> <label> <run> <loader>      (loader: none | state:<edit.npz> | agent:<run dir>)
    -> appends to results/model_parts.jsonl: per part the mean loss and its count, and the total NLL per stay (the summed
       loss of every part over the 600 stays, divided by 600), for the corrected model and the plain model (seed 0)

GPU. Synthetic data only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
COHORT = {"e03": "synth_e03", "n03": "synth_n03"}
key, label, run, loader = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
backend.configure(work_root=str(ROOT / "work"))
backend.configure(cohort=COHORT[key])
if "-pnp" in run:
    backend.setdefault(mhat_file="atoms.mhat_prev.npy")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agent"))

import polars as pl  # noqa: E402
import torch  # noqa: E402

from backend import teacher_forced as op  # noqa: E402
from backend import make_loader  # noqa: E402

orig_load = op.load_model


def load(r: str, how: str):
    model, cfg = orig_load(r)
    if how.startswith("state:"):
        import stateedit
        stateedit.apply(model, stateedit.load(how[6:]), gate_code=stateedit.insulin_code(ROOT / "work" / "bundles" / COHORT[key]))
    elif how.startswith("agent:"):
        import fwd_job
        d = Path(how[6:]).resolve()
        arm = json.loads((d / "state.json").read_text())["arm"]
        sd, _ = fwd_job.edited_state({k: v.detach().cpu() for k, v in model.state_dict().items()}, d / "submission", arm)
        model.load_state_dict(sd)
    return model, cfg


def parts(model, spec, root, stays):
    torch.manual_seed(0)
    loader_ = make_loader(root, "held_out", spec=spec, batch_size=64, shuffle=False, num_workers=0,
                          episode_filter=lambda ep: ep.filter(pl.col("subject_id").is_in(stays)))
    tot, state = {}, None
    with torch.no_grad():
        for b in loader_:
            B = b["bundle_mask"].shape[0]
            if state is None or state["h"].shape[0] != B:
                state = model.init_state(B, torch.device("cuda"))
            b = {k: v.cuda() for k, v in b.items()}
            o, state = model(b, state)
            for name, (s, c) in o["parts"].items():
                a = tot.setdefault(name, [0.0, 0])
                a[0] += float(s)
                a[1] += int(c)
    return tot


def main() -> None:
    op.load_model = lambda r: load(r, loader)
    model, cfg, root, ids, by_id, cut_map, spec, _ = op.setup(run)
    cuts = pl.read_parquet(op.oe.out_dir() / "cuts_held_out.parquet")
    stays = sorted(cuts["subject_id"].to_list())
    plain_run = f"synth_{key}-m-T1-full-long-s0"
    op.load_model = lambda r: load(r, "none")
    plain, _, _, _, _, _, spec0, _ = op.setup(plain_run)
    p_edit, p_plain = parts(model, spec, root, stays), parts(plain, spec0, root, stays)
    common = sorted(set(p_edit) & set(p_plain))
    skip = {"gate_target"}                                        # a target count, not a loss
    rec = {"world": key, "label": label, "run": run, "loader": loader, "n_stays": len(stays), "parts": {}}
    for k in common:
        if k in skip or p_plain[k][1] == 0:
            continue
        rec["parts"][k] = {"plain": round(p_plain[k][0] / p_plain[k][1], 6), "edited": round(p_edit[k][0] / p_edit[k][1], 6),
                           "n": p_plain[k][1]}
    rec["total_nll_per_stay"] = {"plain": round(sum(p_plain[k][0] for k in rec["parts"]) / len(stays), 4),
                                 "edited": round(sum(p_edit[k][0] for k in rec["parts"]) / len(stays), 4)}
    rec["total_nll_per_stay"]["change"] = round(rec["total_nll_per_stay"]["edited"] - rec["total_nll_per_stay"]["plain"], 4)
    rec["extra_parts_edited"] = sorted(set(p_edit) - set(p_plain))
    with (ROOT / "results" / "model_parts.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1), flush=True)


if __name__ == "__main__":
    main()
