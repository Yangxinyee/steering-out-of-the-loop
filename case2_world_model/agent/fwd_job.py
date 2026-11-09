"""One forward pass of the agent's forward tool (agent/forward.py), as its own process on the workstation GPU, outside the
sandbox. It reads the agent's edits from the run's /work but runs none of the agent's code.

    python agent/fwd_job.py <job.json>
job.json: {"request": {...}, "key": "e03" | "n03", "run": <plain model run>, "mode": "white" | "black", "work": <run's /work>,
           "out": <the pass's folder in /work>, "seed": 0}
Writes <out>/summary.json ({"ok": true, ...} or {"ok": false, "error": ...}) and the result files.

Request types (the stays must be in the split; train and tuning only):
  factual    teacher-forced on the whole record: per reading of the four targets, the predicted mean and the NLL of its
             fine bin; summary: value NLL per value (all readings), and per event its NLL and mean absolute error
  dose_edit  the held-out readout's rollout (the companion code's rollout: hour-12 cut, readings hidden from hour 13 and
             drawn by the model, 20 paths with common random numbers) on these stays, for all four edits: per stay-hour
             the predicted shift; summary: per edit, the mean shift over the stay-hours after the drug ran
  states     the state after each bundle: [out (192), h slots 0-3 (4 x 192), z posterior mean (32)] = 992 numbers
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import backend  # the companion paper's code (agent/backend.py)

ROOT = Path(__file__).resolve().parents[1]
COHORT = {"e03": "synth_e03", "n03": "synth_n03"}
TARGET_NAMES = {"glucose_finger": "SOURCE/chartevents/225664", "glucose_lab": "OMOP/3004501",
                "map_arterial": "SOURCE/chartevents/220052", "map_cuff": "SOURCE/chartevents/220181"}
EDIT_NAMES = {"insulin_x2": "insulin x2 -> glucose_finger", "insulin_half": "insulin x1/2 -> glucose_finger",
              "ne_x2": "norepinephrine x2 -> map_arterial", "ne_half": "norepinephrine x1/2 -> map_arterial"}


class JobError(Exception):
    """a problem with the request or the edits (reported to the agent, the pass not counted)"""


def setup_env(key: str) -> None:
    backend.configure(work_root=str(ROOT / "work"))
    backend.configure(cohort=COHORT[key])


def edited_state(sd: dict, work: Path, mode: str) -> tuple[dict, list[str]]:
    """the plain model's weights with the agent's edits applied: white box, /work/edited/<name>.safetensors replace tensors;
    black box, the operations of /work/ops.json applied in order"""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    sd = {k: v.clone() for k, v in sd.items()}
    touched = []
    if mode == "white":
        d = work / "edited"
        for f in sorted(d.glob("*.safetensors")) if d.is_dir() else []:
            name = f.name[: -len(".safetensors")]
            if name not in sd:
                raise JobError(f"edited file for an unknown tensor: {name}")
            t = load_file(str(f))["w"]
            if tuple(t.shape) != tuple(sd[name].shape):
                raise JobError(f"edited {name} has shape {tuple(t.shape)}, the model's is {tuple(sd[name].shape)}")
            if not torch.isfinite(t).all():
                raise JobError(f"edited {name} has NaN or infinite values")
            sd[name] = t.to(sd[name].dtype)
            touched.append(name)
        return sd, touched
    p = work / "ops.json"
    ops = json.loads(p.read_text()) if p.exists() else []
    for i, op in enumerate(ops):
        name = op.get("name")
        if name not in sd:
            raise JobError(f"operation {i}: unknown tensor {name!r}")
        w = sd[name].double()
        if op["op"] == "scale":
            w = w * float(op["c"])
        elif op["op"] == "rank1":
            v = torch.from_numpy(np.load(work / op["vector"])).double().reshape(-1)
            side = op.get("side", "out")
            n = w.shape[0] if side == "out" else (w.shape[1] if w.dim() == 2 else None)
            if n is None or v.numel() != n:
                raise JobError(f"operation {i}: vector of length {v.numel()} does not fit side {side!r} of {name} {tuple(w.shape)}")
            v = v / v.norm().clamp_min(1e-30)
            a = float(op["alpha"])
            w = w + a * (torch.outer(v, v @ w) if w.dim() == 2 and side == "out" else
                         torch.outer(w @ v, v) if side == "in" else v * (v @ w))
        else:
            raise JobError(f"operation {i}: unknown op {op['op']!r}")
        if not torch.isfinite(w).all():
            raise JobError(f"operation {i} gives NaN or infinite values in {name}")
        sd[name] = w.to(sd[name].dtype)
        touched.append(name)
    return sd, sorted(set(touched))


def load_models(run: str, which: list[str], work: Path, mode: str):
    import torch

    from backend import ModelConfig, Vocab
    from backend import WorldModel
    saved = torch.load(ROOT / "work" / "runs" / run / "model.pt", map_location="cpu", weights_only=False)
    out, touched = {}, []
    for m in which:
        sd = saved["model"]
        if m == "edited":
            sd, touched = edited_state(sd, work, mode)
        model = WorldModel(ModelConfig(**saved["config"]), Vocab(**saved["vocab"])).cuda().eval()
        model.load_state_dict(sd)
        model.keep_probs = True
        out[m] = model
    return out, touched


def episodes(split: str, stays: list[int]):
    import polars as pl

    from backend import evaluation as oe
    cuts = pl.read_parquet(oe.out_dir() / f"cuts_{split}.parquet")
    t = oe.cuts(split).select("bundle_lo", "t_cut")
    cuts = cuts.join(t, on="bundle_lo", how="inner")
    have = set(cuts["subject_id"].to_list())
    missing = [s for s in stays if s not in have]
    if missing:
        raise JobError(f"{len(missing)} stay ids are not {split} stays with an hour-12 cut: {missing[:10]}")
    return cuts.filter(pl.col("subject_id").is_in(stays)).sort("bundle_lo")


def factual(models: dict, split: str, stays: list[int], out: Path) -> dict:
    import numpy as np
    import polars as pl
    import torch

    from backend import evaluation as oe
    from backend import BundleStore, WindowSpec, make_loader
    from backend import load_actions
    root = oe.store_root()
    ids = oe.code_ids()
    name_of = {ids[c]: n for n, c in TARGET_NAMES.items() if c in ids}
    store = BundleStore(root, split)
    med, iqr, _ = store.norm["value"]
    cuts = episodes(split, stays)
    t_icu = dict(zip(cuts["subject_id"].to_list(), (cuts["t_cut"] - 12 * 3600).to_list(), strict=True))
    cen = {n: np.asarray(json.loads((root / "vocab" / "target_centres.json").read_text())[c], dtype=np.float64)
           for n, c in TARGET_NAMES.items()}
    any_model = next(iter(models.values()))
    spec = WindowSpec(window=128, effect_fields=False, **load_actions(oe.COHORT, root))
    summary = {}
    for m, model in models.items():
        loader = make_loader(root, split, spec=spec, batch_size=64, shuffle=False, num_workers=0,
                             episode_filter=lambda ep: ep.filter(pl.col("subject_id").is_in(stays)))
        rows, tot, n, state = [], 0.0, 0, None
        with torch.no_grad():
            for b in loader:
                B, K = b["bundle_mask"].shape
                if state is None or state["h"].shape[0] != B:
                    state = model.init_state(B, torch.device("cuda"))
                b = {k: v.cuda() for k, v in b.items()}
                o, state = model(b, state)
                s_, c_ = o["parts"]["value_bin"]
                tot, n = tot + float(s_), n + int(c_)
                att = o["attributes"]["value_bin"]
                atom = att["atom"]
                code = b["atom_code"][atom].cpu().numpy()
                lane = (b["atom_bundle"][atom] // K).cpu().numpy()
                slot = (b["atom_bundle"][atom] % K).cpu().numpy()
                t = b["bundle_t"].cpu().numpy()[lane, slot]
                subj = b["lane_subject"].cpu().numpy()[lane]
                z = b["atom_value"][atom].float().cpu().numpy()
                probs = att["probs"][:, 1:].float().cpu().numpy()
                nll = att["nll"].float().cpu().numpy()
                for i in range(len(atom)):
                    nm = name_of.get(int(code[i]))
                    if nm is None:
                        continue
                    pr = probs[i] / max(probs[i].sum(), 1e-12)
                    rows.append((int(subj[i]), round((int(t[i]) - t_icu[int(subj[i])]) / 3600, 4), nm,
                                 float(med[code[i]] + z[i] * iqr[code[i]]), float(pr @ cen[nm]), float(nll[i])))
        f = pl.DataFrame(rows, schema=["stay_id", "hour", "event", "observed", "predicted_mean", "nll"], orient="row")
        f.write_csv(out / f"factual_{m}.csv", float_precision=4)
        by = f.group_by("event").agg(pl.len().alias("n"), pl.col("nll").mean().round(4).alias("nll"),
                                     (pl.col("observed") - pl.col("predicted_mean")).abs().mean().round(3).alias("mae")).sort("event")
        summary[m] = {"value_nll_per_value": round(tot / max(n, 1), 5), "n_values": n,
                      "by_event": {r["event"]: {"n": r["n"], "nll": r["nll"], "mean_abs_error": r["mae"]} for r in by.iter_rows(named=True)},
                      "file": f"factual_{m}.csv"}
    del any_model
    return summary


def dose_edit(models: dict, split: str, stays: list[int], out: Path, seed: int) -> dict:
    import json as _j

    import numpy as np
    import polars as pl

    from backend import rollout as er
    from backend import evaluation as oe
    from backend import BundleStore, WindowSpec
    from backend import HideObservations
    from backend import load_actions
    root, ids = oe.store_root(), oe.code_ids()
    cuts = episodes(split, stays)
    cut_map = dict(zip(cuts["bundle_lo"].to_list(), cuts["cut_bundle"].to_list(), strict=True))
    spec = WindowSpec(window=128, effect_fields=False, **load_actions(oe.COHORT, root))
    store = BundleStore(root, split)
    bt = np.asarray(store.a["bundle_t"]).astype(np.int64)
    hide_cut = {int(lo): int(lo + np.searchsorted(bt[lo:hi], t + 3600, side="left")) for lo, hi, t in
                zip(cuts["bundle_lo"], cuts["bundle_hi"], cuts["t_cut"], strict=True)}
    eps = store.episodes.filter(pl.col("bundle_lo").is_in(cuts["bundle_lo"].to_list()))
    ds = HideObservations(store, spec, eps, cut_bundle=hide_cut, span_seconds=12 * 3600)
    seq_of_lo = {int(lo): s for s, lo in enumerate(ds.seq_core_lo)}
    centres = {n: np.asarray(_j.loads((root / "vocab" / "target_centres.json").read_text())[oe.TARGETS[n][0]], dtype=np.float64)
               for n in er.TARGETS if oe.TARGETS[n][0] in ids}
    summary = {}
    for m, model in models.items():
        prepared = [er.Prepared(ds, int(ds.first_window_of_seq[seq_of_lo[int(lo)]]), int(cut_map[int(lo)]), int(t), ids)
                    for lo, t in zip(cuts["bundle_lo"], cuts["t_cut"], strict=True)]
        rows = []
        for b in range(0, len(prepared), er.STAYS):
            rows += er.rollout(model, store, prepared[b : b + er.STAYS], centres, 1000 + seed)
        f = pl.DataFrame(rows).rename({"subject": "stay_id"}).select("stay_id", "hour", "edit", "shift", "shift_se", "factual", "reached")
        f.write_csv(out / f"dose_edit_{m}.csv", float_precision=4)
        res = {}
        for e, desc in EDIT_NAMES.items():
            g = f.filter((pl.col("edit") == e) & pl.col("reached"))
            res[e] = {"what": desc, "mean_shift": round(float(g["shift"].mean()), 3) if g.height else None,
                      "n_stay_hours": g.height, "n_stays": g["stay_id"].n_unique()}
        summary[m] = {"edits": res, "file": f"dose_edit_{m}.csv",
                      "note": "shift = predicted mean outcome under the edit minus under the record, hours 13-24; the mean is "
                              "over stay-hours after the drug had run from hour 12 on (the 'reached' rows)"}
    return summary


def states(models: dict, split: str, stays: list[int], out: Path, which: str) -> dict:
    import numpy as np
    import polars as pl
    import torch

    from backend import evaluation as oe
    from backend import WindowSpec, make_loader
    from backend import load_actions
    root, ids = oe.store_root(), oe.code_ids()
    ins, ne = ids["SOURCE/inputevents/223258"], ids["OMOP/1321341"]
    cuts = episodes(split, stays)
    t_icu = dict(zip(cuts["subject_id"].to_list(), (cuts["t_cut"] - 12 * 3600).to_list(), strict=True))
    spec = WindowSpec(window=128, effect_fields=False, **load_actions(oe.COHORT, root))
    summary = {}
    for m, model in models.items():
        rec, zrec = [], []
        orig = model.core.step

        def step(state, bundle, dt, active, extra=(), n_loops=None, _orig=orig):
            new, o = _orig(state, bundle, dt, active, extra, n_loops)
            rec.append((o.detach().float(), new["h"].detach().float()))
            return new, o

        model.core.step = step
        zdim = model.cfg.z_dim
        hook = model.z_post.register_forward_hook(lambda mod, i, o: zrec.append(o.detach().float()[..., :zdim]))
        loader = make_loader(root, split, spec=spec, batch_size=64, shuffle=False, num_workers=0,
                             episode_filter=lambda ep: ep.filter(pl.col("subject_id").is_in(stays)))
        feats, keys, state = [], [], None
        with torch.no_grad():
            for b in loader:
                B, K = b["bundle_mask"].shape
                if state is None or state["h"].shape[0] != B:
                    state = model.init_state(B, torch.device("cuda"))
                b = {k: v.cuda() for k, v in b.items()}
                rec.clear()
                zrec.clear()
                _, state = model(b, state)
                o = torch.stack([r[0] for r in rec], 1)
                hs = torch.stack([r[1] for r in rec], 1).flatten(2)
                zs = torch.stack(zrec, 1)
                x = torch.cat([o, hs, zs], -1).cpu().numpy()
                mask, t, subj = b["bundle_mask"].cpu().numpy(), b["bundle_t"].cpu().numpy(), b["lane_subject"].cpu().numpy()
                code, kind = b["atom_code"].cpu().numpy(), b["atom_kind"].cpu().numpy()
                bundle = b["atom_bundle"].cpu().numpy()
                rate = b["atom_rate"].float().cpu().numpy() if "atom_rate" in b else np.zeros(len(code))
                has = {c: np.zeros(B * K, bool) for c in (ins, ne)}
                for c in (ins, ne):
                    sel = (code == c) & (kind == 0)
                    has[c][bundle[sel]] = True
                n_atoms = np.bincount(bundle, minlength=B * K)
                for lb in range(B):
                    for k in np.flatnonzero(mask[lb]):
                        g = lb * K + k
                        hi, hn = bool(has[ins][g]), bool(has[ne][g])
                        if which == "insulin" and not hi or which == "norepinephrine" and not hn or which == "drug" and not (hi or hn):
                            continue
                        keys.append((int(subj[lb]), round((int(t[lb, k]) - t_icu[int(subj[lb])]) / 3600, 4), hi, hn, int(n_atoms[g])))
                        feats.append(x[lb, k])
        hook.remove()
        model.core.step = orig
        del rate
        a = np.stack(feats).astype(np.float32) if feats else np.zeros((0, 992), np.float32)
        np.savez_compressed(out / f"states_{m}.npz", x=a, stay_id=np.array([k[0] for k in keys], np.int64),
                            hour=np.array([k[1] for k in keys]), has_insulin=np.array([k[2] for k in keys]),
                            has_norepinephrine=np.array([k[3] for k in keys]), n_atoms=np.array([k[4] for k in keys]))
        summary[m] = {"file": f"states_{m}.npz", "rows": int(a.shape[0]), "layout": "x[:, 0:192] = out, x[:, 192 + 192*s : 384 + 192*s] = "
                      "slot s of h (s = 0..3), x[:, 960:992] = z posterior mean; one row per bundle, after the bundle",
                      "other_arrays": ["stay_id", "hour", "has_insulin", "has_norepinephrine", "n_atoms"]}
    return summary


def edit_hash(work: Path, mode: str) -> str:
    h = hashlib.sha256()
    files = sorted((work / "edited").glob("*.safetensors")) if mode == "white" and (work / "edited").is_dir() else []
    if mode == "black" and (work / "ops.json").exists():
        files = [work / "ops.json"] + sorted({work / o["vector"] for o in json.loads((work / "ops.json").read_text()) if o.get("vector")})
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:10] if files else "none"


def main() -> None:
    job = json.loads(Path(sys.argv[1]).read_text())
    out, work = Path(job["out"]), Path(job["work"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        setup_env(job["key"])
        import torch
        req = job["request"]
        which = ["original", "edited"] if req["model"] == "both" else [req["model"]]
        models, touched = load_models(job["run"], which, work, job["mode"])
        stays = [int(s) for s in req["stays"]]
        if req["type"] == "factual":
            res = factual(models, req["split"], stays, out)
        elif req["type"] == "dose_edit":
            res = dose_edit(models, req["split"], stays, out, int(job.get("seed", 0)))
        else:
            res = states(models, req["split"], stays, out, req.get("bundles", "all"))
        summary = {"ok": True, "type": req["type"], "split": req["split"], "n_stays": len(stays), "results": res,
                   "edited_tensors": touched if "edited" in which else None, "edit_hash": edit_hash(work, job["mode"]),
                   "seconds": round(time.time() - t0, 1), "gpu_peak_mib": round(torch.cuda.max_memory_reserved() / 2**20)}
    except JobError as e:
        summary = {"ok": False, "error": str(e), "kind": "request"}
    except Exception as e:  # noqa: BLE001
        summary = {"ok": False, "error": f"{type(e).__name__}: {e}", "kind": "internal",
                   "traceback": traceback.format_exc()[-3000:]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
