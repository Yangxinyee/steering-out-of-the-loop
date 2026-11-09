"""Every loss part of the plain model, original vs an agent's edited weights, teacher-forced on the 600 held-out stays.
agent_eval.py checks only the value head; a submission that edits many tensors (claude_a: 71, including the time, count,
status and delay heads) can move the other parts too.

    python scripts/agent_parts.py <agent run dir> [<agent run dir> ...]
GPU, the companion code's Python. Writes results/agent_parts.jsonl. Synthetic data only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
runs = [Path(a).resolve() for a in sys.argv[1:]]
keys = {json.loads((d / "state.json").read_text())["world"] for d in runs}
assert len(keys) == 1, "one world per call (the cohort is set before the imports)"
key = keys.pop()
COHORT = {"e03": "synth_e03", "n03": "synth_n03"}
backend.configure(work_root=str(ROOT / "work"))
backend.configure(cohort=COHORT[key])
sys.path.insert(0, str(ROOT / "agent"))

import polars as pl  # noqa: E402
import torch  # noqa: E402

import fwd_job  # noqa: E402
from backend import evaluation as oe  # noqa: E402
from backend import WindowSpec, make_loader  # noqa: E402
from backend import load_actions  # noqa: E402

root = oe.store_root()
cuts = pl.read_parquet(oe.out_dir() / "cuts_held_out.parquet")
stays = sorted(cuts["subject_id"].to_list())
spec = WindowSpec(window=128, effect_fields=False, **load_actions(oe.COHORT, root))


def parts(model):
    torch.manual_seed(0)                                       # the same random numbers for original and edited
    loader = make_loader(root, "held_out", spec=spec, batch_size=64, shuffle=False, num_workers=0,
                         episode_filter=lambda ep: ep.filter(pl.col("subject_id").is_in(stays)))
    tot, state = {}, None
    with torch.no_grad():
        for b in loader:
            B = b["bundle_mask"].shape[0]
            if state is None or state["h"].shape[0] != B:
                state = model.init_state(B, torch.device("cuda"))
            b = {k: v.cuda() for k, v in b.items()}
            o, state = model(b, state)
            for name, (s, c) in o["parts"].items():
                a = tot.setdefault(name, [0.0, 0])
                a[0] += float(s)
                a[1] += int(c)
    return {k: (s / c if c else None, c) for k, (s, c) in tot.items()}


for d in runs:
    st = json.loads((d / "state.json").read_text())
    models, _ = fwd_job.load_models(st["model_run"], ["original", "edited"], d / "submission", st["arm"])
    p = {m: parts(models[m]) for m in ("original", "edited")}
    rec = {"run": d.name, "world": key, "arm": st["arm"], "split": "held_out", "parts": {}}
    for name in sorted(p["original"]):
        o, n = p["original"][name]
        e, _ = p["edited"][name]
        if o is None or e is None:                             # a part with no items in these stays
            continue
        rec["parts"][name] = {"original": round(o, 5), "edited": round(e, 5), "change": round(e - o, 5), "n": n}
        print(f"{d.name:10s} {name:22s} n={n:>8d}  {o:9.5f} -> {e:9.5f}  ({e - o:+.5f})", flush=True)
    with (ROOT / "results" / "agent_parts.jsonl").open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    del models
    torch.cuda.empty_cache()
