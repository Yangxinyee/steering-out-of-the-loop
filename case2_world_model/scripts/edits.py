"""State edits for the amnesic test and control 1 (steering), built on CPU from the probe's saved states.

    python scripts/edits.py amnesic <e03|n03> <run>   # LEACE: erase all linear information about s beyond the dose
                                                      # -> work/edits/<key>__<run>__amnesic.npz, __random.npz (same form)
    python scripts/edits.py steer <e03|n03> <run>     # the mean-difference direction of s within each sliding-scale level
                                                      # -> work/edits/<key>__<run>__steer_a<alpha>.npz for alpha in ALPHAS
    python scripts/edits.py nll <e03|n03> <run> <edit.npz|none> <split>   # factual value NLL per value (teacher-forced, CPU)

The states are [out, h] (960) after each insulin bundle, from scripts/probe.py extract (train: first 3,000 stays; held-out:
600). Results are appended to results/edits.jsonl.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
backend.configure(work_root=str(ROOT / "work"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

import probe  # noqa: E402
import stateedit  # noqa: E402

EDITS = ROOT / "work" / "edits"
LOG = ROOT / "results" / "edits.jsonl"
ALPHAS = (1.0, 2.0, 4.0, 8.0)
MAX_DIRS, STOP_MARGIN = 64, 0.02


def log(rec: dict) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


def data(key: str, run: str):
    tr, xtr = probe.table(key, "train", run)
    ho, xho = probe.table(key, "held_out", run)
    d = lambda f: np.c_[np.log(f["ins"].to_numpy()), np.log(f["ins"].to_numpy()) ** 2]  # noqa: E731
    return tr, xtr[:, :stateedit.D_STATE].astype(np.float64), ho, xho[:, :stateedit.D_STATE].astype(np.float64), d(tr), d(ho)


def r2_state_dose(xtr, xho, dtr, dho, ytr, yho) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(np.c_[dtr, xtr], ytr)
    p = m.predict(np.c_[dho, xho])
    return float(1 - ((yho - p) ** 2).sum() / ((yho - yho.mean()) ** 2).sum())


def leace(xc: np.ndarray, z: np.ndarray, p: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """LEACE (Belrose et al., arXiv 2306.03819) for a scalar concept z on centred states xc: the rank-1 eraser
    x' = x - L R^T (x - mu) with R = W p, L = W^+ p, W the (pseudo-)whitening, p the unit whitened cross-covariance.
    With ``p`` given (a random unit vector in whitened space) it is the same-form random control."""
    cov = xc.T @ xc / len(xc)
    lam, V = np.linalg.eigh(cov)
    keep = lam > lam.max() * float(os.environ.get("LEACE_FLOOR", "1e-6"))
    V, lam = V[:, keep], lam[keep]
    W, Wp = (V / np.sqrt(lam)) @ V.T, (V * np.sqrt(lam)) @ V.T
    if p is None:
        p = W @ (xc.T @ z / len(xc))
    else:
        p = V @ (V.T @ p)                                     # inside the support
    p = p / np.linalg.norm(p)
    return W @ p, Wp @ p


def amnesic(key: str, run: str) -> None:
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr, xtr, ho, xho, dtr, dho = data(key, run)
    ytr, yho = tr["s"].to_numpy(), ho["s"].to_numpy()
    mu = xtr.mean(0)
    base = float(1 - ((yho - LinearRegression().fit(dtr, ytr).predict(dho)) ** 2).sum() / ((yho - yho.mean()) ** 2).sum())
    z = ytr - LinearRegression().fit(dtr, ytr).predict(dtr)   # the part of s the dose does not explain (linearly)
    xc = xtr - mu
    rng = np.random.default_rng(0)
    rec = {"kind": "amnesic_leace", "key": key, "run": run, "floor": os.environ.get("LEACE_FLOOR", "1e-6"), "tag": os.environ.get("LEACE_TAG", ""),
           "dose_only_r2": round(base, 4),
           "before": round(r2_state_dose(xtr, xho, dtr, dho, ytr, yho), 4)}
    var = float((xc**2).sum(1).mean())
    EDITS.mkdir(parents=True, exist_ok=True)
    # the variance-matched random control: a whitened direction inside the eigen-band whose variance is near the amnesic
    # eraser's, so that it moves about as much of the state (the plain random direction moves almost nothing)
    R0, L0 = leace(xc, z)
    moved = float((((xc @ R0)[:, None] * L0[None, :]) ** 2).sum(1).mean())
    lam, V = np.linalg.eigh(xc.T @ xc / len(xc))
    band = np.flatnonzero((lam > 0.5 * moved) & (lam < 2.0 * moved))
    band = band if band.size else np.array([int(np.argmin(np.abs(lam - moved)))])
    p_matched = V[:, band] @ rng.standard_normal(band.size)
    for name, p in (("amnesic", None), ("random", rng.standard_normal(stateedit.D_STATE)), ("random_matched", p_matched)):
        R, L = leace(xc, z, p)
        E = lambda x: x - ((x - mu) @ R)[:, None] * L[None, :]  # noqa: E731
        xtr2, xho2 = E(xtr), E(xho)
        rec[name] = {"r2_linear_after": round(r2_state_dose(xtr2, xho2, dtr, dho, ytr, yho), 4),
                     "share_of_state_variance_moved": round(float(((xtr2 - xtr) ** 2).sum(1).mean() / var), 5)}
        mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0))
        mlp.fit(np.c_[dtr, xtr2], ytr)
        pm = mlp.predict(np.c_[dho, xho2])
        rec[name]["r2_mlp_after"] = round(float(1 - ((yho - pm) ** 2).sum() / ((yho - yho.mean()) ** 2).sum()), 4)
        tag = os.environ.get("LEACE_TAG", "")
        stateedit.save_oblique(EDITS / f"{key}__{run}__{name}{tag}.npz", mu, R, L, np.array([-1.0]))
    log(rec)


def steer(key: str, run: str) -> None:
    tr, xtr, ho, xho, dtr, dho = data(key, run)
    mu = xtr.mean(0)
    diffs, weights = [], []
    for lvl in sorted(tr["scale"].unique().to_list()):
        sel = (tr["scale"] == lvl).to_numpy()
        s = tr["s"].to_numpy()[sel]
        if sel.sum() < 200:
            continue
        lo, hi = np.quantile(s, [0.25, 0.75])
        x = xtr[sel]
        diffs.append(x[s >= hi].mean(0) - x[s <= lo].mean(0))
        weights.append(sel.sum())
    u = np.average(np.stack(diffs), axis=0, weights=weights)
    u /= np.linalg.norm(u)
    # the probe's direction: ridge of s on the state, both with the dose features regressed out (INLP's first direction);
    # the raw mean difference is dominated by the state's high-variance directions, so it is kept as the plan's version
    # and the probe direction is added beside it
    from sklearn.linear_model import LinearRegression, RidgeCV
    res = lambda a: a - LinearRegression().fit(dtr, a).predict(dtr)  # noqa: E731
    w = RidgeCV(alphas=np.logspace(-2, 6, 17), fit_intercept=False).fit(res(xtr - mu), res(tr["s"].to_numpy())).coef_
    w /= np.linalg.norm(w)
    EDITS.mkdir(parents=True, exist_ok=True)
    for a in ALPHAS:
        stateedit.save(EDITS / f"{key}__{run}__steer_a{a:g}.npz", mu, u[:, None], np.array([a]))
        stateedit.save(EDITS / f"{key}__{run}__steerp_a{a:g}.npz", mu, w[:, None], np.array([a]))
    s_ho, ld = ho["s"].to_numpy(), np.log(ho["ins"].to_numpy())
    rec = {"kind": "steer_direction", "key": key, "run": run, "levels": len(diffs)}
    for nm, v in (("mean_diff", u), ("probe", w)):
        proj = (xho - mu) @ v
        rec[nm] = {"corr_with_s_held_out": round(float(np.corrcoef(proj, s_ho)[0, 1]), 4),
                   "corr_with_log_dose_held_out": round(float(np.corrcoef(proj, ld)[0, 1]), 4),
                   "sd_of_projection": round(float(proj.std()), 4)}
    rec["cosine_mean_diff_probe"] = round(float(u @ w), 4)
    log(rec)


def nll(key: str, run: str, edit: str, split: str) -> None:
    import torch

    from backend import WindowSpec, make_loader
    from backend import ModelConfig, Vocab
    from backend import WorldModel
    from backend import load_actions

    torch.set_num_threads(int(os.environ.get("PROBE_THREADS", "8")))
    cohort = probe.WORLDS[key][1]
    root = ROOT / "work" / "bundles" / cohort
    saved = torch.load(ROOT / "work" / "runs" / run / "model.pt", map_location="cpu", weights_only=False)
    model = WorldModel(ModelConfig(**saved["config"]), Vocab(**saved["vocab"])).eval()
    model.load_state_dict(saved["model"])
    if edit != "none":
        stateedit.apply(model, stateedit.load(edit), gate_code=stateedit.insulin_code(root))
    spec = WindowSpec(window=128, effect_fields=bool(model.cfg.effect_pathway), **load_actions(cohort, root))
    loader = make_loader(root, split, spec=spec, batch_size=64, shuffle=False, num_workers=0)
    tot, n, state = 0.0, 0, None
    with torch.no_grad():
        for b in loader:
            B = b["bundle_mask"].shape[0]
            if state is None or state["h"].shape[0] != B:
                state = model.init_state(B, torch.device("cpu"))
            out, state = model(b, state)
            s_, c_ = out["parts"]["value_bin"]
            tot, n = tot + float(s_), n + int(c_)
    log({"kind": "nll", "key": key, "run": run, "edit": Path(edit).name if edit != "none" else "none", "split": split,
         "value_nll_per_value": round(tot / n, 5), "values": n})


if __name__ == "__main__":
    {"amnesic": lambda: amnesic(sys.argv[2], sys.argv[3]), "steer": lambda: steer(sys.argv[2], sys.argv[3]),
     "nll": lambda: nll(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])}[sys.argv[1]]()
