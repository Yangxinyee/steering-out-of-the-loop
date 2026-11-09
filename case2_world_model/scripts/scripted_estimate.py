"""The scripted baseline's estimator: each drug's direct effect per unit of rate at lags 1-12 h, from
the training records alone, by two-stage least squares with the dose's deviation from the expected dose as the instrument.

Written once and run unchanged on both worlds (no thresholds of the simulator's policy are written in):
- the instrument: a given dose minus the mean dose given in the same cell of what the clinician saw (leave-one-out; cells
  in deviations()); doses not given, and doses that followed a missing reading or a reading on a cell edge, carry none;
- outcome Y(t) on the rates R(t-1..t-12), instrumented by the deviations D(t-1..t-12), with hour-of-stay fixed effects and
  readings no dose in the window can have moved (hour 0, and the outcome before hour t - 12) as controls; standard errors
  clustered by stay. Later doses enter as regressors, so the effect of a dose is net of the doses that follow it.

    python3 scripts/scripted_estimate.py <e03|n03>   -> results/scripted/estimate_<world>.json
The kernel used downstream is lags 1-8 of the 12-lag fit (lags 9-12 are kept in the regression as nuisance terms; they drift
in the no-effect world). Also writes the implied tuning-stay readouts of the four queried edits (sustained x2, x1/2 from hour 12, pooled over the
stay-hours after the drug ran from hour 12, as the agents' dose_edit tool pools them). CPU, synthetic data only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
H, LAGS = 26, 12
EVENTS = ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff")


def panel(csv: Path) -> dict[str, np.ndarray]:
    """per stay x hour since ICU admission (0..25): the four readings (NaN if none) and the two rates (0 if none)"""
    f = pl.read_csv(csv, columns=["stay_id", "hour", "event", "value", "rate"])
    f = f.filter(pl.col("hour") >= 0).with_columns(pl.col("hour").floor().cast(pl.Int64).alias("h")).filter(pl.col("h") < H)
    stays = np.sort(f["stay_id"].unique().to_numpy())
    idx = {s: i for i, s in enumerate(stays)}
    out = {"stay": stays}
    for e in EVENTS:
        g = f.filter(pl.col("event") == e)
        a = np.full((len(stays), H), np.nan)
        a[[idx[s] for s in g["stay_id"].to_list()], g["h"].to_numpy()] = g["value"].to_numpy()
        out[e] = a
    for e, k in (("insulin", "I"), ("norepinephrine", "N")):
        g = f.filter((pl.col("event") == e) & (pl.col("rate") > 0))
        a = np.zeros((len(stays), H))
        np.add.at(a, ([idx[s] for s in g["stay_id"].to_list()], g["h"].to_numpy()), g["rate"].to_numpy())
        out[k] = a
    return out


def loo_mean(cell: np.ndarray, y: np.ndarray) -> np.ndarray:
    """leave-one-out mean of y within each cell (NaN for a cell of one)"""
    _, inv, cnt = np.unique(cell, return_inverse=True, return_counts=True)
    tot = np.bincount(inv, weights=y)
    n = cnt[inv]
    return np.where(n > 1, (tot[inv] - y) / np.maximum(n - 1, 1), np.nan)


def cells_of(v: np.ndarray, width: float, n: int) -> np.ndarray:
    c = np.clip(np.floor(v / width), 0, n - 1)
    return np.where(np.isfinite(v), c, n).astype(np.int64)          # n = missing


def on_edge(v: np.ndarray, width: float) -> np.ndarray:
    return np.isfinite(v) & (np.abs(v / width - np.round(v / width)) < 1e-9)


def deviations(p: dict) -> dict[str, np.ndarray]:
    """each given dose minus the mean dose given in the same cell of what the clinician saw (leave-one-out); 0 (no instrument)
    where no dose was given, where the reading the dose followed is missing, or where that reading lies on a cell edge (a
    rounded reading there does not say which side of the edge the clinician saw)
    - insulin: cell = the hour's finger-stick and laboratory glucose, 10 mg/dL each (missing is its own cell); the reading the
      dose followed: finger-stick, else laboratory;
    - norepinephrine: cell = the previous hour's rate (its 0.01 grid) x the hour's MAP, 5 mmHg (arterial, else cuff, and which)
    """
    I, N = p["I"], p["N"]
    n = I.shape[0]
    gf, gl = p["glucose_finger"], p["glucose_lab"]
    cell_i = cells_of(gf, 10, 80) * 100 + cells_of(gl, 10, 80)
    ok_i = (I > 0) & ~on_edge(np.where(np.isfinite(gf), gf, gl), 10)
    m = np.where(np.isfinite(p["map_arterial"]), p["map_arterial"], p["map_cuff"])
    which = np.where(np.isfinite(p["map_arterial"]), 0, np.where(np.isfinite(p["map_cuff"]), 1, 2))
    nprev = np.concatenate([np.zeros((n, 1)), N[:, :-1]], axis=1)
    cell_n = np.round(nprev * 100).astype(np.int64) * 200 + cells_of(m, 5, 40) * 3 + which
    ok_n = (N > 0) & (which < 2) & ~on_edge(m, 5)
    out = {}
    for k, R, cell, ok in (("I", I, cell_i, ok_i), ("N", N, cell_n, ok_n)):
        D = np.zeros_like(R)
        D[ok] = R[ok] - loo_mean(cell[ok], R[ok])
        out[k] = np.nan_to_num(D)
        print(f"  {k}: instrument at {ok.sum()} of {(R > 0).sum()} dosed stay-hours", flush=True)
    return out


def controls(p: dict, outcome: str, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """readings no dose in the 12-hour window can have moved: the four readings at hour 0 (taken before the first dose) and
    the outcome's last reading at or before hour t - 13; each with a missing indicator"""
    cols = []
    for e in EVENTS:
        v = p[e][s, 0]
        cols += [np.nan_to_num(v - np.nanmean(v)), np.isfinite(v).astype(float)]
    y = p[outcome]
    last = np.full(len(t), np.nan)
    for back in range(13, 26):
        h = t - back
        ok = (h >= 0) & ~np.isfinite(last)
        last[ok] = y[s[ok], h[ok]]
    cols += [np.nan_to_num(last - np.nanmean(last)), np.isfinite(last).astype(float)]
    return np.stack(cols, axis=1)


def lagged(a: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """[rows, LAGS]: a[s, t - L] for L = 1..LAGS (0 before admission)"""
    out = np.zeros((len(t), LAGS))
    for L in range(1, LAGS + 1):
        ok = t - L >= 0
        out[ok, L - 1] = a[s[ok], t[ok] - L]
    return out


def tsls(p: dict, D: dict, drug: str, outcome: str) -> dict:
    y = p[outcome]
    s, t = np.nonzero(np.isfinite(y) & (np.arange(H)[None, :] >= 1) & (np.arange(H)[None, :] <= 24))
    Y = y[s, t]
    X, Z = lagged(p[drug], t, s), lagged(D[drug], t, s)

    W = np.concatenate([np.eye(26)[t], controls(p, outcome, t, s)], axis=1)       # hour-of-stay fixed effects + controls
    WtW = np.linalg.pinv(W.T @ W)

    def demean(a):
        return a - W @ (WtW @ (W.T @ a))
    Yd, Xd, Zd = demean(Y[:, None])[:, 0], demean(X), demean(Z)
    G = np.linalg.solve(Zd.T @ Zd, Zd.T @ Xd)            # first stage
    Xh = Zd @ G
    A = Xh.T @ Xd
    beta = np.linalg.solve(A, Xh.T @ Yd)
    u = Yd - Xd @ beta
    S = np.zeros((len(np.unique(s)), LAGS))
    _, inv = np.unique(s, return_inverse=True)
    np.add.at(S, inv, Xh * u[:, None])
    Ai = np.linalg.inv(A)
    V = Ai @ (S.T @ S) @ Ai.T
    w = np.ones(LAGS)
    # first-stage strength: partial F of each lag's own deviation (one regressor), the smallest over lags
    F = []
    for L in range(LAGS):
        z, x = Zd[:, L], Xd[:, L]
        b = (z @ x) / (z @ z)
        r = x - b * z
        F.append(float(b * b * (z @ z) / (r @ r / (len(x) - 1))))
    return {"psi": beta.round(4).tolist(), "se": np.sqrt(np.diag(V)).round(4).tolist(), "sum": round(float(w @ beta), 3),
            "sum_se": round(float(np.sqrt(w @ V @ w)), 3), "n_rows": int(len(Y)), "n_stays": int(len(np.unique(s))),
            "min_first_stage_F": round(min(F), 1), "_V": V.tolist()}


KERNEL_LAGS = 8


def kernel(pair: dict) -> np.ndarray:
    """the kernel used downstream: lags 1..KERNEL_LAGS of the fit, zero beyond"""
    k = np.zeros(LAGS)
    k[:KERNEL_LAGS] = pair["psi"][:KERNEL_LAGS]
    return k


def implied(p: dict, drug: str, outcome: str, psi: np.ndarray, factor: float, start: int = 12) -> tuple[float, int]:
    """the estimate's readout of a sustained edit x factor from hour `start`: (f - 1) * sum_L psi_L R(t - L) over doses at
    hours >= start, pooled over stay-hours t in start+1..start+12 with the outcome read and the drug run at an hour in [start, t)"""
    R, y = p[drug], p[outcome]
    vals = []
    for t in range(start + 1, min(start + 13, 25)):
        ran = (R[:, start:t] > 0).any(axis=1)
        ok = np.isfinite(y[:, t]) & ran
        sh = np.zeros(R.shape[0])
        for L in range(1, LAGS + 1):
            if t - L >= start:
                sh += psi[L - 1] * (factor - 1) * R[:, t - L]
        vals.append(sh[ok])
    v = np.concatenate(vals)
    return float(v.mean()), int(len(v))


def main(key: str) -> None:
    data = ROOT / "agent_data" / key / "data"
    tr = panel(data / "train.csv")
    D = deviations(tr)
    est = {"world": key, "lags_h": LAGS, "pairs": {}}
    for drug, outcome in (("I", "glucose_finger"), ("I", "glucose_lab"), ("N", "map_arterial"), ("N", "map_cuff")):
        est["pairs"][f"{drug}->{outcome}"] = r = tsls(tr, D, drug, outcome)
        V = np.array(r.pop("_V"))[:KERNEL_LAGS, :KERNEL_LAGS]
        r["sum8"], r["sum8_se"] = round(float(np.sum(r["psi"][:KERNEL_LAGS])), 3), round(float(np.sqrt(V.sum())), 3)
        print(f"{key} {drug}->{outcome}: lags 1-8 sum {r['sum8']} (se {r['sum8_se']}); 12 lags {r['sum']} (se {r['sum_se']}), psi {r['psi'][:6]}..., n {r['n_rows']}, "
              f"min F {r['min_first_stage_F']}", flush=True)
    tu = panel(data / "tuning.csv")
    est["tuning_targets"] = {}
    for e, (drug, outcome, f) in {"insulin_x2": ("I", "glucose_finger", 2.0), "insulin_half": ("I", "glucose_finger", 0.5),
                                  "ne_x2": ("N", "map_arterial", 2.0), "ne_half": ("N", "map_arterial", 0.5)}.items():
        m, n = implied(tu, drug, outcome, kernel(est["pairs"][f"{drug}->{outcome}"]), f)
        est["tuning_targets"][e] = {"mean_shift": round(m, 3), "n_stay_hours": n}
    print(key, "tuning targets:", est["tuning_targets"], flush=True)
    out = ROOT / "results" / "scripted"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"estimate_{key}.json").write_text(json.dumps(est, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
