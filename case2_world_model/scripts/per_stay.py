"""Per-stay agreement on the queried x2 questions : each method's per-stay mean shift against the
truth, with two references -- the plain model, and a dose-only predictor that knows nothing about the patient: the record's
own doses from hour 12 on through an estimated effect kernel (Claude's from the records, or the scripted estimator's).

Per method and question: correlation (stay bootstrap), RMSE, and the RMSE left after the best single rescaling of the
method's per-stay shifts (scale-free RMSE: what remains once the overall size is put right).
    python scripts/per_stay.py   -> results/per_stay.json. CPU, synthetic data only.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import generalize_summary as gs  # noqa: E402

T13 = ROOT / "work" / "runs" / "table13" / gs.WORLD["e03"][0]
METHODS = [("plain model", "id_es_register"), ("steering, mean difference", "steer_a0.25"), ("steering, probe direction", "steerp_a32"),
           ("fine-tuning", "ft_pnp"), ("Codex, black box", "agent_openai_c"), ("Claude, black box", "agent_claude_c"),
           ("Codex, white box", "agent_openai_a"), ("Claude, white box", "agent_claude_a"),
           ("scripted install, Claude's estimate", "scriptC"), ("fine-tuning sized to Claude's estimate", "ftC"),
           ("scripted install, scripted estimate", "scriptS")]
QUESTIONS = {"insulin_x2": ("INS", "I"), "ne_x2": ("NE", "N")}
CLAUDE_PSI = {"I": [-2.0, -2.3, -2.0, -1.6, -1.1, -0.9, -0.6, -0.4], "N": [3.4, 2.6, 1.75, 0.9, 0.4, 0.2]}


def stats(x: np.ndarray, y: np.ndarray) -> dict:
    a = float(x @ y / (x @ x)) if x @ x > 0 else float("nan")
    return {"corr": gs.boot_corr(x, y), "rmse": round(float(np.sqrt(np.mean((x - y) ** 2))), 3),
            "best_scale": round(a, 3), "rmse_scale_free": round(float(np.sqrt(np.mean((a * x - y) ** 2))), 3)}


def main() -> None:
    s = gs.sim.simulate(gs.WORLD["e03"][1], gs.N, gs.SEED)
    row_of = {int(e): i for i, e in enumerate(s["subject"])}
    scripted = json.loads((ROOT / "results" / "scripted" / "estimate_e03.json").read_text())["pairs"]
    psi_s = {"I": scripted["I->glucose_finger"]["psi"][:8], "N": scripted["N->map_arterial"]["psi"][:8]}
    out = {"n_stays": {}, "truth_sd": {}, "methods": {}}
    ref = pl.read_parquet(T13 / "shifts_id_es_register_s0.parquet")
    for q, (col, d) in QUESTIONS.items():
        g = ref.filter((pl.col("edit") == q) & pl.col("reached")).join(gs.truth("e03", q, 12), on=["episode", "hour"], how="left")
        dose = {}
        for name, psi in (("Claude's kernel", CLAUDE_PSI[d]), ("scripted kernel", psi_s[d])):
            v = []
            for e, h in zip(g["episode"].to_list(), g["hour"].to_list(), strict=True):
                r = s[col][row_of[int(e)]]
                v.append(sum(p * r[gs.sim.PRE + h - L] for L, p in enumerate(psi, start=1) if h - L >= 12))
            dose[name] = np.array(v)
        gd = g.with_columns(*[pl.Series(f"dose_{k}", v) for k, v in dose.items()])
        st = gd.group_by("episode").agg(pl.col("true_shift").mean().alias("t"), *[pl.col(f"dose_{k}").mean().alias(k) for k in dose]).sort("episode")
        y = st["t"].to_numpy()
        out["n_stays"][q], out["truth_sd"][q] = len(y), round(float(np.std(y)), 3)
        for k in dose:
            out["methods"].setdefault(f"dose only ({k})", {})[q] = stats(st[k].to_numpy(), y)
        for label, method in METHODS:
            f = T13 / f"shifts_{method}_s0.parquet"
            if not f.is_file():
                continue
            m = pl.read_parquet(f).filter((pl.col("edit") == q) & pl.col("reached")).join(gs.truth("e03", q, 12), on=["episode", "hour"], how="left")
            ms = m.group_by("episode").agg(pl.col("shift").mean().alias("m"), pl.col("true_shift").mean().alias("t")).sort("episode")
            ms = ms.join(st.select("episode"), on="episode", how="inner")
            out["methods"].setdefault(label, {})[q] = stats(ms["m"].to_numpy(), ms["t"].to_numpy())
    (ROOT / "results" / "per_stay.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["methods"].items():
        print(k, {q: (r["corr"][0], r["rmse"], r["rmse_scale_free"], r["best_scale"]) for q, r in v.items()})


if __name__ == "__main__":
    main()
