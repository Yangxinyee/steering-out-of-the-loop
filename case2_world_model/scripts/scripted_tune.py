"""Size the scripted baseline on the tuning stays, as the agents sized their edits: the agents' own
dose_edit tool code (agent/fwd_job.py) on the 400 tuning stays, one gain per drug, until the four readouts match the estimate.

    python scripts/scripted_tune.py <variant>
variants (VARIANTS below): which world, which model the pathway goes into (plain | ft), which estimate (claude: the effect
Claude estimated from the records and matched its edit to, its lag profile and its four tuning-stay readouts; script: the
fixed estimator scripts/scripted_estimate.py, its readouts computed on the same stay-hours the tool pools).
Per round: build -> dose_edit on the 400 tuning stays -> per drug the least-squares gain on its two edits from the readouts
(R = R0 + g u: the first round from the model at gain 0, later rounds the secant through the last two). At most 4 rounds; stops
when both x2 readouts are within 3% of the target. Then the agents' factual check (value NLL per reading, edited minus plain).
-> work/runs/<run>/ (model.pt of the last round), results/scripted/tune_<variant>.json. GPU. Synthetic data only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "agent"))

#: Claude's estimate (agent_runs/claude_a/work/notes.md): lag profiles per unit of rate (insulin lags 1-8 on finger-stick
#: glucose, norepinephrine lags 1-6 on arterial MAP) and the readouts it computed from them on the 400 tuning stays, which its
#: final edit was matched to (final_answer.md, "True (from records)")
CLAUDE = {"kernels": {"I": [-2.0, -2.3, -2.0, -1.6, -1.1, -0.9, -0.6, -0.4], "N": [3.4, 2.6, 1.75, 0.9, 0.4, 0.2]},
          "targets": {"insulin_x2": -13.415, "insulin_half": 6.707, "ne_x2": 4.474, "ne_half": -2.237}}
#: the plain model's tuning readouts (Claude's tool, all 400 tuning stays, original weights): R at gain 0 for source plain
PLAIN_TUNING = {"insulin_x2": -1.80, "insulin_half": 0.82, "ne_x2": -0.23, "ne_half": -0.56}
VARIANTS = {
    "scriptC_e03": {"key": "e03", "source": "plain", "estimate": "claude", "run": "synth_e03-m-T1-full-scriptC-pnp-s0"},
    "ftC_e03": {"key": "e03", "source": "ft", "estimate": "claude", "run": "synth_e03-m-T1-full-ftC-pnp-s0"},
    "scriptS_e03": {"key": "e03", "source": "plain", "estimate": "script", "run": "synth_e03-m-T1-full-scriptS-pnp-s0"},
    "scriptS_n03": {"key": "n03", "source": "plain", "estimate": "script", "run": "synth_n03-m-T1-full-scriptS-pnp-s0"},
}
DRUG_EDITS = {"I": ("insulin_x2", "insulin_half"), "N": ("ne_x2", "ne_half")}
variant = sys.argv[1]
V = VARIANTS[variant]
key = V["key"]
backend.configure(mhat_file="atoms.mhat_prev.npy")
import fwd_job  # noqa: E402

fwd_job.setup_env(key)
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402

from backend import dataset as dsmod  # noqa: E402
from backend import ModelConfig, Vocab  # noqa: E402
from backend import WorldModel  # noqa: E402
from scripted_build import build, plain_run  # noqa: E402
from scripted_estimate import LAGS, panel  # noqa: E402

_WS = dsmod.WindowSpec


def _spec(*a, **k):                     # the tool's passes, with the fields the pathway reads (rates, clinician model)
    k["effect_fields"] = True
    return _WS(*a, **k)


dsmod.WindowSpec = _spec
STAYS = sorted(int(s) for b in json.loads((ROOT / "agent_runs" / "claude_a" / "work" / "batches.json").read_text()).values() for s in b)
assert len(STAYS) == 400


def load(run: str):
    saved = torch.load(ROOT / "work" / "runs" / run / "model.pt", map_location="cpu", weights_only=False)
    m = WorldModel(ModelConfig(**saved["config"]), Vocab(**saved["vocab"])).cuda().eval()
    m.load_state_dict(saved["model"])
    m.keep_probs = True
    return m


def estimate() -> tuple[dict, dict | None]:
    if V["estimate"] == "claude":
        return CLAUDE["kernels"], CLAUDE["targets"]
    e = json.loads((ROOT / "results" / "scripted" / f"estimate_{key}.json").read_text())
    return {"I": e["pairs"]["I->glucose_finger"]["psi"][:8], "N": e["pairs"]["N->map_arterial"]["psi"][:8]}, None


def row_targets(csv: Path, kernels: dict) -> dict:
    """the estimate's readout on exactly the stay-hours the tool pools: (f - 1) sum_L psi_L R(t - L) over doses from hour 12"""
    p = panel(ROOT / "agent_data" / key / "data" / "tuning.csv")
    row_of = {s: i for i, s in enumerate(p["stay"].tolist())}
    f = pl.read_csv(csv).filter(pl.col("reached"))
    out = {}
    for e, (drug, fac) in {"insulin_x2": ("I", 2.0), "insulin_half": ("I", 0.5), "ne_x2": ("N", 2.0), "ne_half": ("N", 0.5)}.items():
        psi = np.zeros(LAGS)
        psi[: len(kernels[drug])] = kernels[drug]
        g = f.filter(pl.col("edit") == e)
        R = p[drug]
        vals = []
        for s, h in zip(g["stay_id"].to_list(), g["hour"].to_list(), strict=True):
            i, t = row_of[int(s)], int(h)
            vals.append(sum(psi[L - 1] * (fac - 1) * R[i, t - L] for L in range(1, LAGS + 1) if t - L >= 12))
        out[e] = float(np.mean(vals))
    return out


def readouts(run: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    model = load(run)
    res = fwd_job.dose_edit({"m": model}, "tuning", STAYS, out, 0)
    del model
    torch.cuda.empty_cache()
    return {e: v["mean_shift"] for e, v in res["m"]["edits"].items()}


def main() -> None:
    kernels, targets = estimate()
    log = {"variant": variant, **V, "kernels": kernels, "rounds": []}
    rec_path = ROOT / "results" / "scripted" / f"tune_{variant}.json"
    t0 = time.time()
    g = {"I": 1.0, "N": 1.0}
    hist = []          # (gains, readouts)
    if V["source"] == "plain":
        hist.append(({"I": 0.0, "N": 0.0}, dict(PLAIN_TUNING)))
    for rnd in range(4):
        build(key, V["run"], V["source"], kernels, g)
        R = readouts(V["run"], ROOT / "work" / "scripted" / variant / f"round{rnd}")
        if targets is None:
            targets = row_targets(ROOT / "work" / "scripted" / variant / f"round{rnd}" / "dose_edit_m.csv", kernels)
            log["targets_from"] = "estimate on the tool's reached rows, round 0"
        hist.append((dict(g), R))
        log["rounds"].append({"gains": dict(g), "readouts": R, "targets": targets, "minutes": round((time.time() - t0) / 60, 1)})
        log["targets"] = targets
        rec_path.write_text(json.dumps(log, indent=1))
        print(f"round {rnd}: gains {g} readouts {R} targets {targets}", flush=True)
        ok = all(abs(R[e] - targets[e]) <= 0.03 * abs(targets[e]) + 0.05 for e in ("insulin_x2", "ne_x2"))
        if ok:
            break
        # source ft, first round: no measured point at gain 0, so the readout is taken as proportional to the gain
        (g0, R0), (g1, R1) = (hist[-2] if len(hist) > 1 else ({"I": 0.0, "N": 0.0}, {e: 0.0 for e in R})), hist[-1]
        new = {}
        for d, es in DRUG_EDITS.items():
            u = np.array([(R1[e] - R0[e]) for e in es]) / (g1[d] - g0[d]) if g1[d] != g0[d] else None
            if u is None or not np.isfinite(u).all() or (u ** 2).sum() < 1e-9:
                new[d] = g1[d]
                continue
            T = np.array([targets[e] for e in es])
            base = np.array([R1[e] for e in es]) - g1[d] * u
            new[d] = float(np.clip((u @ (T - base)) / (u @ u), -50, 50))
        g = new
    else:
        print("not converged in 4 rounds; the last round's model is kept", flush=True)
        log["converged"] = False
    log.setdefault("converged", True)
    log["final_gains"] = hist[-1][0]
    if log["converged"] is False or hist[-1][0] != g:
        build(key, V["run"], V["source"], kernels, hist[-1][0])        # the model on disk is the last measured one
    # the agents' factual check: value NLL per reading on the 400 tuning stays, edited minus plain
    fo = ROOT / "work" / "scripted" / variant / "factual"
    fo.mkdir(parents=True, exist_ok=True)
    res = fwd_job.factual({"edited": load(V["run"]), "original": load(plain_run(key))}, "tuning", STAYS, fo)
    log["factual"] = {"edited": res["edited"]["value_nll_per_value"], "original": res["original"]["value_nll_per_value"],
                      "diff": round(res["edited"]["value_nll_per_value"] - res["original"]["value_nll_per_value"], 5),
                      "by_event": {m: res[m]["by_event"] for m in res}}
    log["minutes"] = round((time.time() - t0) / 60, 1)
    rec_path.write_text(json.dumps(log, indent=1))
    print("factual", log["factual"]["diff"], "minutes", log["minutes"], flush=True)


if __name__ == "__main__":
    main()
