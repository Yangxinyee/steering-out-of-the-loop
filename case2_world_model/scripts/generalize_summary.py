"""Score the generalization rollouts (scripts/generalize.py) and the original held-out rollouts against the simulator's truth.

    python scripts/generalize_summary.py            (the companion code's Python; CPU)
    -> results/generalize.json, results/generalize.md

Truth: the simulator's replay of the same 600 held-out stays (seed 303) with the edited drug's logged doses times the factor
from the start hour on and the clinician no longer reacting -- synth_world.truth with other factors and start hours. The
replay at factor 2 from hour 12 reproduces truth_held_out.parquet (checked). Hours are labelled 13..24 relative to the start.

Per model and edit: the readout (mean model shift / mean true shift over reached stay-hours; stays resampled 1,000 times),
as table13 computes it. From the original rollouts (the four questions the agents saw), also: the readout over the first and
the last four hours, the hourly profile, and the per-stay agreement (correlation across stays of the stay's mean shift with
its true mean shift, and the per-stay error relative to the truth's spread).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
from backend import sim  # noqa: E402

WORLD = {"e03": ("s+1_b1_e0.3", sim.World(1, 1.0, 0.3)), "n03": ("s+0_b1_e0.3", sim.World(0, 1.0, 0.3))}
N, SEED = 600, 303
DRUG = {"insulin": "INS", "norepinephrine": "NE"}
FACTOR = {"insulin_x2": ("insulin", 2.0, "G"), "insulin_half": ("insulin", 0.5, "G"), "ne_x2": ("norepinephrine", 2.0, "M"),
          "ne_half": ("norepinephrine", 0.5, "M"), "insulin_x0": ("insulin", 0.0, "G"), "insulin_x1.5": ("insulin", 1.5, "G"),
          "insulin_x3": ("insulin", 3.0, "G"), "insulin_x2_lab": ("insulin", 2.0, "G"), "ne_x0": ("norepinephrine", 0.0, "M"),
          "ne_x1.5": ("norepinephrine", 1.5, "M"), "ne_x3": ("norepinephrine", 3.0, "M"), "ne_x2_cuff": ("norepinephrine", 2.0, "M")}
#: (label, world, table13 method of the original rollout, generalize.py tag)
MODELS = [("plain model", "e03", "id_es_register", "plain"), ("steering, mean difference", "e03", "steer_a0.25", "steer"),
          ("steering, probe direction", "e03", "steerp_a32", "steerp"), ("fine-tuning", "e03", "ft_pnp", "ft"),
          ("erasure (diagnostic)", "e03", "amnesic", "amnesic"),
          ("Codex, black box", "e03", "agent_openai_c", "openai_c"), ("Claude, black box", "e03", "agent_claude_c", "claude_c"),
          ("Codex, white box", "e03", "agent_openai_a", "openai_a"), ("Claude, white box", "e03", "agent_claude_a", "claude_a"),
          ("plain model", "n03", None, "plain"), ("fine-tuning", "n03", None, "ft"),
          ("Codex, white box", "n03", None, "openai_b"), ("Claude, white box", "n03", None, "claude_b"),
          # the information-matched baselines without an agent (scripts/scripted_tune.py)
          ("scripted install, Claude's estimate", "e03", "scriptC", "scriptC"),
          ("scripted install, Claude's estimate, stopped dose as rate 0", "e03", None, "scriptCz"),
          ("fine-tuning sized to Claude's estimate", "e03", "ftC", "ftC"),
          ("scripted install, scripted estimate", "e03", "scriptS", "scriptS"),
          ("scripted install, scripted estimate", "n03", None, "scriptS")]


def boot_ratio(a, b, subj, n=1000, seed=0):
    s_, inv = np.unique(subj, return_inverse=True)
    ta, tb = np.bincount(inv, a), np.bincount(inv, b)
    pick = np.random.default_rng(seed).integers(0, len(s_), size=(n, len(s_)))
    r = ta[pick].sum(1) / tb[pick].sum(1)
    return [round(float(ta.sum() / tb.sum()), 4), round(float(np.quantile(r, 0.025)), 4), round(float(np.quantile(r, 0.975)), 4)]


def boot_mean(v, subj, n=1000, seed=0):
    s_, inv = np.unique(subj, return_inverse=True)
    tot, cnt = np.bincount(inv, v), np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(s_), size=(n, len(s_)))
    b = tot[pick].sum(1) / cnt[pick].sum(1)
    return [round(float(tot.sum() / cnt.sum()), 4), round(float(np.quantile(b, 0.025)), 4), round(float(np.quantile(b, 0.975)), 4)]


def boot_corr(x, y, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    r = [np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return [round(float(np.corrcoef(x, y)[0, 1]), 4), round(float(np.nanquantile(r, 0.025)), 4), round(float(np.nanquantile(r, 0.975)), 4)]


_TRUTH: dict = {}


def truth(world_key: str, edit: str, start: int) -> pl.DataFrame:
    """episode, hour label (13..), true shift of the hidden quantity"""
    k = (world_key, edit, start)
    if k not in _TRUTH:
        name, w = WORLD[world_key]
        if (world_key, "s") not in _TRUTH:
            _TRUTH[(world_key, "s")] = sim.simulate(w, N, SEED)
        s = _TRUTH[(world_key, "s")]
        drug, f, q = FACTOR[edit]
        r = sim.replay(w, N, SEED, s, {DRUG[drug]: f}, start)
        d = r["edited"][q] - r["factual"][q]
        parts = [pl.DataFrame({"episode": s["subject"].astype(np.int64), "hour": 12 + (h - start), "true_shift": d[:, sim.PRE + h]})
                 for h in range(start + 1, min(start + 13, sim.HOURS))]
        _TRUTH[k] = pl.concat(parts)
    return _TRUTH[k]


def check_truth() -> None:
    ref = pl.read_parquet(ROOT / "work" / "synth" / WORLD["e03"][0] / "truth_held_out.parquet").filter(pl.col("edit") == "insulin_x2")
    mine = truth("e03", "insulin_x2", 12).join(ref.select("episode", "hour", pl.col("true_shift").alias("ref")), on=["episode", "hour"])
    assert mine.height == ref.height and float((mine["true_shift"] - mine["ref"]).abs().max()) < 1e-9, "truth replay mismatch"


def score(rows: pl.DataFrame, world_key: str, edit: str, start: int) -> dict | None:
    # the same 12-hour window as table13 (hours 13..24 relative to the start): a start before hour 12 hides one more hour
    g = rows.filter((pl.col("edit") == edit) & pl.col("reached") & (pl.col("hour") <= 24))
    if g.height == 0:
        return None
    g = g.join(truth(world_key, edit, start), on=["episode", "hour"], how="left")
    assert g["true_shift"].null_count() == 0, (edit, start)
    v, t, subj = g["shift"].to_numpy(), g["true_shift"].to_numpy(), g["subject"].to_numpy()
    r = {"n_stays": g["episode"].n_unique(), "n_stay_hours": g.height, "model_shift": boot_mean(v, subj), "true_shift": boot_mean(t, subj)}
    if world_key == "e03" and np.abs(t).max() > 0:
        r["ratio"] = boot_ratio(v, t, subj)
    return r


def detail(rows: pl.DataFrame, edit: str) -> dict:
    """hourly profile, early/late horizons and per-stay agreement for one original question"""
    g = rows.filter((pl.col("edit") == edit) & pl.col("reached")).join(truth("e03", edit, 12), on=["episode", "hour"], how="left")
    v, t, subj, hr = g["shift"].to_numpy(), g["true_shift"].to_numpy(), g["subject"].to_numpy(), g["hour"].to_numpy()
    out = {"hourly": {}, "horizon": {}}
    for h in range(13, 25):
        m = hr == h
        if m.sum() > 20:
            out["hourly"][h] = {"model": boot_mean(v[m], subj[m]), "truth": boot_mean(t[m], subj[m])}
    for name, (a, b) in {"hours 13-16": (13, 16), "hours 21-24": (21, 24)}.items():
        m = (hr >= a) & (hr <= b)
        out["horizon"][name] = boot_ratio(v[m], t[m], subj[m])
    st = g.group_by("subject").agg(pl.col("shift").mean().alias("m"), pl.col("true_shift").mean().alias("t")).sort("subject")
    x, y = st["m"].to_numpy(), st["t"].to_numpy()
    out["per_stay"] = {"n": len(x), "corr": boot_corr(x, y), "rmse": round(float(np.sqrt(np.mean((x - y) ** 2))), 3),
                       "truth_sd": round(float(np.std(y)), 3)}
    return out


def main() -> None:
    check_truth()
    res = {}
    t13 = {"e03": ROOT / "work" / "runs" / "table13" / WORLD["e03"][0]}
    gen = {k: ROOT / "work" / "runs" / "general" / WORLD[k][0] for k in WORLD}
    for label, wk, method, tag in MODELS:
        key = f"{wk}:{label}"
        res[key] = {"world": wk, "label": label}
        if method is not None and (t13[wk] / f"shifts_{method}_s0.parquet").is_file():
            rows = pl.read_parquet(t13[wk] / f"shifts_{method}_s0.parquet")
            res[key]["original"] = {e: score(rows, wk, e, 12) for e in ("insulin_x2", "insulin_half", "ne_x2", "ne_half")}
            res[key]["detail"] = {e: detail(rows, e) for e in ("insulin_x2", "ne_x2")}
        for design in ("doses", "starts"):
            files = sorted(gen[wk].glob(f"{tag}_s0_{design}*.parquet"))       # a whole run, or its pieces (_c0, _h6, ...)
            if not files:
                continue
            rows = pl.concat([pl.read_parquet(f) for f in files]).unique(subset=["episode", "hour", "edit", "start"], keep="first")
            for start in sorted(rows["start"].unique().to_list()):
                sub = rows.filter(pl.col("start") == start)
                for e in sorted(sub["edit"].unique().to_list()):
                    res[key].setdefault(design, {})[f"{e}@{start}"] = score(sub, wk, e, start)
    (ROOT / "results" / "generalize.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: {d: {e: (v or {}).get("ratio", (v or {}).get("model_shift")) for e, v in r.get(d, {}).items()}
                          for d in ("original", "doses", "starts") if d in r} for k, r in res.items()}, indent=1))


if __name__ == "__main__":
    main()
