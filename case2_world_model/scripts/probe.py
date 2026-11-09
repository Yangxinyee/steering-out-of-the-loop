"""CPU linear probe: is the surprise dose readable from the plain event-stream model's state after the dose bundle?

Target s = log(insulin rate) - log(the sliding-scale rate for the clinician's latest glucose), recomputed from the simulator
with the split seeds (exact). Rows: insulin-running hours 1..24 (hour 0 is the admission rate, no decision). Features: the
plain model's state after the bundle at h:05 that carries hour h's doses -- the core's output (192), its 4 slots (768) and
the posterior mean of z (32); z enters the state as its posterior mean at evaluation, so the state is deterministic.

Rulers on the same rows: dose only (log rate, its square); the record's ceiling (log rate and the clinician model's
prediction mhat_prev, LightGBM on the observed history, cross-fitted on train). Fit on the first N_TRAIN training stays,
report R^2 on the 600 held-out stays with a patient bootstrap. td100: the ceiling only (its store has its own vocabulary,
so the main-world model is not run on it; no td100 plain model was trained).

    python scripts/probe.py extract <world_key> <run> [n_train]     # world_key: e03 | n03
    python scripts/probe.py fit_part <world_key> <run|none>          # one model (or the record rulers alone)
    python scripts/probe.py fit                                      # merge -> results/probe.json
Synthetic data only; everything is read from and written to this project's work/ (the companion code's work root).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))
import backend  # noqa: E402  (the companion paper's code)
backend.configure(work_root=str(ROOT / "work"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from backend import sim  # noqa: E402

WORK = ROOT / "work"
OUT = WORK / "probe"
WORLDS = {"e03": ("s+1_b1_e0.3", "synth_e03", sim.World(1, 1.0, 0.3)), "n03": ("s+0_b1_e0.3", "synth_n03", sim.World(0, 1.0, 0.3)),
          "td100": ("s+1_b1_e0.3_td100", "synth_e03_td100", sim.World(1, 1.0, 0.3))}
SPLITS = {"train": (30_000, 101), "tuning": (400, 202), "held_out": (600, 303)}
INS_CODE = "SOURCE/inputevents/223258"
T0 = int((datetime(2150, 1, 1) - datetime(1970, 1, 1)).total_seconds())
DOSE_S = 5 * 60
N_TRAIN = 3000
THREADS = int(os.environ.get("PROBE_THREADS", "16"))


def truth(world: sim.World, split: str) -> pl.DataFrame:
    """(subject, hour) -> insulin rate, the sliding-scale rate, s, log expected rate; insulin-running hours 1..24."""
    n, seed = SPLITS[split]
    s = sim.simulate(world, n, seed)
    g = np.where(np.isnan(s["FSG"]), s["GLU"], s["FSG"])
    last = g.copy()
    for t in range(1, last.shape[1]):
        last[:, t] = np.where(np.isnan(last[:, t]), last[:, t - 1], last[:, t])
    scale = np.select([last < 150, last < 180, last < 220, last < 260, last < 300], [0.0, 1.0, 2.0, 3.0, 4.0], 6.0)
    rows = []
    for h in range(1, sim.HOURS):
        c = sim.PRE + h
        ins, sc = s["INS"][:, c], scale[:, c]
        ok = (ins > 0) & (sc > 0) & np.isfinite(last[:, c])
        rows.append(pl.DataFrame({"subject": s["subject"][ok].astype(np.int64), "hour": h, "ins": ins[ok], "scale": sc[ok],
                                  "s": np.log(ins[ok]) - np.log(sc[ok]), "log_expected": np.log(sc[ok])}))
    return pl.concat(rows)


def mhat(cohort: str, split: str) -> pl.DataFrame:
    """The clinician model's prediction of each insulin atom's log rate (atoms.mhat_prev.npy), by (subject, hour)."""
    d = WORK / "bundles" / cohort / split
    ids = pl.read_parquet(d / "atoms.event_id.parquet")["event_id"].str.split("|")
    f = pl.DataFrame({"subject": ids.list.get(0).cast(pl.Int64), "code": ids.list.get(1), "t": ids.list.get(2).cast(pl.Int64),
                      "mhat": np.load(d / "atoms.mhat_prev.npy"), "rate": np.load(d / "atoms.rate.npy").astype(np.float64)})
    f = f.filter(pl.col("code") == INS_CODE).with_columns(hour=((pl.col("t") - T0 - DOSE_S) // 3600).cast(pl.Int32))
    return f.select("subject", "hour", "mhat", "rate")


def extract(key: str, run: str, n_train: int = N_TRAIN) -> None:
    import torch

    from backend import WindowSpec, make_loader
    from backend import ModelConfig, Vocab
    from backend import WorldModel
    from backend import load_actions

    torch.set_num_threads(THREADS)
    _, cohort, world = WORLDS[key]
    root = WORK / "bundles" / cohort
    # "random:<run>": the same architecture and vocabulary as <run>, weights at their random initialisation (seed 0) --
    # the baseline for what any state built from the record carries (Li et al. 2023 skip targets linear in the input)
    random_init = run.startswith("random:")
    src = run.split(":", 1)[1] if random_init else run
    saved = torch.load(WORK / "runs" / src / "model.pt", map_location="cpu", weights_only=False)
    torch.manual_seed(0)
    model = WorldModel(ModelConfig(**saved["config"]), Vocab(**saved["vocab"])).eval()
    if not random_init:
        model.load_state_dict(saved["model"])
    assert not model.cfg.effect_pathway, "the probe is for the plain model"
    model.z_samples = 1                                    # the heads' output is not used; one z draw is enough
    rec: list = []
    orig = model.core.step

    def step(state, bundle, dt, active, extra=(), n_loops=None):
        new, out = orig(state, bundle, dt, active, extra, n_loops)
        rec.append((out.detach().float(), new["h"].detach().float()))
        return new, out

    model.core.step = step
    zdim = model.cfg.z_dim
    zrec: list = []
    model.z_post.register_forward_hook(lambda m, i, o: zrec.append(o.detach().float()[..., :zdim]))
    spec = WindowSpec(window=128, effect_fields=False, **load_actions(cohort, root))
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "held_out"):
        tr = truth(world, split)
        want = {(int(a), int(b)) for a, b in tr.select("subject", "hour").iter_rows()}
        flt = (lambda ep: ep.sort("subject_id").head(n_train)) if split == "train" else None
        loader = make_loader(root, split, spec=spec, batch_size=64, shuffle=False, num_workers=0, episode_filter=flt)
        feats, keys, t0 = [], [], time.time()
        state = None
        with torch.no_grad():
            for n, batch in enumerate(loader):
                B, K = batch["bundle_mask"].shape
                if state is None or state["h"].shape[0] != B:
                    state = model.init_state(B, torch.device("cpu"))
                rec.clear(); zrec.clear()
                _, state = model(batch, state)
                assert len(rec) == K and len(zrec) == K, (len(rec), len(zrec), K)
                out = torch.stack([r[0] for r in rec], 1)                    # [B, K, D]
                hs = torch.stack([r[1] for r in rec], 1).flatten(2)         # [B, K, slots*D]
                zs = torch.stack(zrec, 1)                                   # [B, K, zdim]
                bt, mask, subj = batch["bundle_t"].numpy(), batch["bundle_mask"].numpy(), batch["lane_subject"].numpy()
                for b in range(B):
                    for k in np.flatnonzero(mask[b]):
                        rel = int(bt[b, k]) - T0
                        if rel % 3600 != DOSE_S:
                            continue
                        key_ = (int(subj[b]), rel // 3600)
                        if key_ in want:
                            keys.append(key_)
                            feats.append(torch.cat([out[b, k], hs[b, k], zs[b, k]]).numpy())
                if n % 10 == 0:
                    print(f"{key} {run} {split} batch {n} rows {len(keys)} {time.time() - t0:.0f}s", flush=True)
        kf = pl.DataFrame({"subject": [a for a, _ in keys], "hour": [b for _, b in keys]}, schema={"subject": pl.Int64, "hour": pl.Int32})
        np.savez_compressed(OUT / f"{key}__{run}__{split}.npz", x=np.stack(feats).astype(np.float32),
                            subject=kf["subject"].to_numpy(), hour=kf["hour"].to_numpy(), d_out=192, d_h=768, d_z=zdim)
        print(f"wrote {key} {run} {split}: {len(keys)} rows of {len(want) if split != 'train' else 'n/a'}; {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------------------------------------------- fit
def r2_boot(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n: int = 1000) -> list[float]:
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rng = np.random.default_rng(0)
    u, inv = np.unique(groups, return_inverse=True)
    idx_of = [np.flatnonzero(inv == i) for i in range(len(u))]
    bs = []
    for _ in range(n):
        ix = np.concatenate([idx_of[i] for i in rng.integers(0, len(u), len(u))])
        yy, pp = y[ix], p[ix]
        bs.append(1 - ((yy - pp) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
    return [round(float(r2), 4), round(float(np.quantile(bs, 0.025)), 4), round(float(np.quantile(bs, 0.975)), 4)]


def table(key: str, split: str, run: str | None) -> tuple[pl.DataFrame, np.ndarray | None]:
    _, cohort, world = WORLDS[key]
    t = truth(world, split).join(mhat(cohort, split), on=["subject", "hour"], how="inner")
    if run is None:
        if split == "train":
            t = t.filter(pl.col("subject").is_in(t["subject"].unique().sort().head(N_TRAIN).implode()))
        return t, None
    z = np.load(OUT / f"{key}__{run}__{split}.npz")
    f = pl.DataFrame({"subject": z["subject"], "hour": z["hour"].astype(np.int32)}).with_row_index("row")
    j = t.join(f, on=["subject", "hour"], how="inner").sort("row")
    return j, z["x"][j["row"].to_numpy()]


def fit_one(key: str, run: str | None) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, xtr = table(key, "train", run)
    ho, xho = table(key, "held_out", run)
    out = {"rows_train": tr.height, "rows_held_out": ho.height, "stays_held_out": ho["subject"].n_unique()}
    g = ho["subject"].to_numpy()

    def lin(cols_tr, cols_ho, y):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 4, 15))).fit(cols_tr, tr[y].to_numpy())
        return r2_boot(ho[y].to_numpy(), m.predict(cols_ho), g)

    def rec(f: pl.DataFrame) -> dict[str, np.ndarray]:
        lr = np.log(f["ins"].to_numpy())
        return {"dose": np.c_[lr, lr**2], "record": np.c_[lr, lr**2, f["mhat"].to_numpy(), lr * f["mhat"].to_numpy()]}

    a, b = rec(tr), rec(ho)
    for y in ("s", "log_expected"):
        out[y] = {"dose_only": lin(a["dose"], b["dose"], y), "record_ceiling": lin(a["record"], b["record"], y)}
        if xtr is not None:
            # the state alone, and the state beside the dose features: the question is whether the state adds the expected
            # dose (the history's part) to what the dose itself already says
            sd_tr, sd_ho = np.c_[a["dose"], xtr], np.c_[b["dose"], xho]
            out[y]["probe_out"] = lin(xtr[:, :192], xho[:, :192], y)
            out[y]["probe_state"] = lin(xtr, xho, y)
            out[y]["probe_state_dose"] = lin(sd_tr, sd_ho, y)
            if y == "s":
                for name, (u, v) in (("probe_state_mlp", (xtr, xho)), ("probe_state_dose_mlp", (sd_tr, sd_ho))):
                    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0))
                    mlp.fit(u, tr[y].to_numpy())
                    out[y][name] = r2_boot(ho[y].to_numpy(), mlp.predict(v), g)
            lo, hi = out[y]["dose_only"][0], out[y]["record_ceiling"][0]
            for name in ("probe_state_dose", "probe_state_dose_mlp"):
                if name in out[y] and hi > lo:
                    out[y][f"gap_closed_{name[6:]}"] = round((out[y][name][0] - lo) / (hi - lo), 3)
    out["s_sd_held_out"] = round(float(ho["s"].std()), 4)
    return out


RUNS = {"e03": [f"synth_e03-m-T1-full-long-s{k}" for k in range(3)] + ["random:synth_e03-m-T1-full-long-s0"],
        "n03": [f"synth_n03-m-T1-full-long-s{k}" for k in range(3)], "td100": []}
PARTS = ROOT / "results" / "probe_parts"


def part_file(key: str, run: str) -> Path:
    """one probe part per world and model; ':' (in "random:<run>") is not portable in file names"""
    return PARTS / f"{key}__{run.replace(':', '_')}.json"


def fit_part(key: str, run: str) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    r = fit_one(key, None if run == "none" else run)
    part_file(key, run).write_text(json.dumps(r, indent=1) + "\n")
    print(key, run, json.dumps(r["s"]), flush=True)


def fit() -> None:
    res = {"what": __doc__.split("\n\n")[0], "target": "s = log(insulin rate) - log(sliding-scale rate); held-out R^2 [95% patient bootstrap]",
           "readable_rule": "the state beside the dose closes at least half the gap from dose_only to record_ceiling (gap_closed >= 0.5)",
           "worlds": {}}
    for key in ("e03", "n03", "td100"):
        res["worlds"][key] = {"record_only": json.loads((PARTS / f"{key}__none.json").read_text())}
        for run in RUNS[key]:
            if part_file(key, run).is_file():
                res["worlds"][key][run] = json.loads(part_file(key, run).read_text())
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "probe.json").write_text(json.dumps(res, indent=1) + "\n")
    f = lambda v: f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]" if isinstance(v, list) else "—"  # noqa: E731
    lines = ["| world | model | dose only | record ceiling | state (linear) | state + dose (linear) | state + dose (MLP) | gap closed (linear / MLP) |", "|---|---|---|---|---|---|---|---|"]
    for key, rows in res["worlds"].items():
        for run, r in rows.items():
            x = r["s"]
            gl, gm = x.get("gap_closed_state_dose"), x.get("gap_closed_state_dose_mlp")
            lines.append(f"| {key} | {run.replace('-m-T1-full-long', '')} | {f(x['dose_only'])} | {f(x['record_ceiling'])} | {f(x.get('probe_state'))} | "
                         f"{f(x.get('probe_state_dose'))} | {f(x.get('probe_state_dose_mlp'))} | {gl if gl is not None else '—'} / {gm if gm is not None else '—'} |")
    (ROOT / "results" / "probe_table.md").write_text("Held-out R^2 of the surprise dose s (in brackets: 95% patient-bootstrap interval)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "extract":
        extract(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else N_TRAIN)
    elif sys.argv[1] == "fit_part":
        fit_part(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "fit":
        fit()
