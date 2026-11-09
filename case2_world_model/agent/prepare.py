"""What the agent is given, built once per world from this project's copies in work/:

    agent_data/<e03|n03>/model/        model.safetensors (the plain model, seed 0, float32), tensors.json (names and shapes),
                                       vocab.json (codes, normalisation, bin edges and centres), ARCHITECTURE.md
    agent_data/<e03|n03>/model_black/  the same without model.safetensors (the black-box arm sees names and shapes only)
    agent_data/<e03|n03>/data/         train.csv (30,000 stays), tuning.csv (400 stays), README.md

Both worlds are written in exactly the same form: the agent is not told which world it is in. No held-out stay, truth,
simulator code or world name is copied. Synthetic data only.

    <the companion code's python> -m agent.prepare
"""
from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
OUT = ROOT / "agent_data"
ASSETS = Path(__file__).resolve().parent / "assets"
WORLDS = {"e03": "synth_e03", "n03": "synth_n03"}
SEED = 0
NAMES = {"SOURCE/chartevents/225664": ("glucose_finger", "mg/dL"), "OMOP/3004501": ("glucose_lab", "mg/dL"),
         "SOURCE/chartevents/220052": ("map_arterial", "mmHg"), "SOURCE/chartevents/220181": ("map_cuff", "mmHg"),
         "SOURCE/inputevents/223258": ("insulin", "units/hour"), "OMOP/1321341": ("norepinephrine", "mcg/kg/min"),
         "SYNTH/admission": ("hospital_admission", None), "SYNTH/icu_stay": ("icu_admission", None)}
DRUGS = ("insulin", "norepinephrine")


def run_name(key: str) -> str:
    return f"{WORLDS[key]}-m-T1-full-long-s{SEED}"


def vocab(key: str) -> dict:
    v = WORK / "bundles" / WORLDS[key] / "vocab"
    codes = pl.read_parquet(v / "codes.parquet")
    kinds = pl.read_parquet(v / "kinds.parquet")
    num = pl.read_parquet(v / "numeric.parquet")
    centres = json.loads((v / "target_centres.json").read_text())
    out = {"codes": [{"id": 0, "name": "(padding)"}] + [{"id": int(r["oov_code_id"]), "name": f"(unknown {r['event_kind']})"}
                                                        for r in kinds.iter_rows(named=True)]}
    for r in codes.iter_rows(named=True):
        name, unit = NAMES[r["code"]]
        out["codes"].append({"id": int(r["code_id"]), "name": name, "event_kind": r["event_kind"], "unit": unit,
                             "is_action": name in DRUGS})
    out["event_kinds"] = {int(r["kind_id"]): r["event_kind"] for r in kinds.iter_rows(named=True)}
    out["atom_kinds"] = {0: "WORLD (the event happened)", 1: "REVEAL (a result becomes known; not used in this data)",
                         2: "END (an infusion record ends)"}
    attr_of = {"value": "value", "rate": "rate", "dose_num": "dose"}
    out["numeric"] = []
    fine = [f"f{i}" for i in range(1, 16)]
    for r in num.iter_rows(named=True):
        out["numeric"].append({"code": NAMES[r["code"]][0], "attribute": attr_of[r["attr"]], "median": r["p50"],
                               "iqr": r["p75"] - r["p25"], "fine_bin_edges": [r[f] for f in fine]})
    out["numeric_note"] = ("z = (x - median) / iqr per code, clipped to [-10, 10]; fine bin b in 1..16 is 1 + the number of "
                           "edges below x (0 = no value). value_bin, rate_bin and dose_bin use these bins.")
    out["value_bin_centres"] = {NAMES[c][0]: cs for c, cs in centres.items()}
    out["value_bin_note"] = ("The value head has 17 classes: 0 = no value, 1..16 = the fine bins. The model's predicted mean of "
                             "a reading is sum over b = 1..16 of p(b) * centre[b - 1].")
    return out


def config(key: str) -> dict:
    saved = torch.load(WORK / "runs" / run_name(key) / "model.pt", map_location="cpu", weights_only=False)
    keep = ("d_model", "n_heads", "ff_mult", "n_prelude", "n_recurrent", "n_coda", "n_loops", "n_slots", "n_memory", "n_time_mix",
            "z_dim", "z_eval_samples", "tau_out_range_s", "tau_max_s")
    return {k: saved["config"][k] for k in keep}, saved["model"]


def events(key: str, split: str) -> pl.DataFrame:
    files = sorted(glob.glob(str(WORK / "cohorts" / WORLDS[key] / "events" / f"split={split}" / "*.parquet")))
    ev = pl.concat([pl.read_parquet(f, columns=["subject_id", "time", "code", "numeric_value", "end_time", "rate", "rate_unit",
                                                 "dose", "unit"]) for f in files])
    icu = ev.filter(pl.col("code") == "SYNTH/icu_stay").group_by("subject_id").agg(pl.col("time").min().alias("t_icu"))
    ev = ev.join(icu, on="subject_id", how="inner")
    hours = lambda c: ((pl.col(c) - pl.col("t_icu")).dt.total_seconds() / 3600.0).round(4)  # noqa: E731
    name = pl.col("code").replace_strict({c: n for c, (n, _) in NAMES.items()})
    out = ev.select(pl.col("subject_id").alias("stay_id"), hours("time").alias("hour"), name.alias("event"),
                    pl.col("numeric_value").cast(pl.Float64).alias("value"), pl.col("rate"), pl.col("rate_unit"),
                    pl.col("dose").cast(pl.Float64).alias("dose"), pl.col("unit").alias("dose_unit"),
                    hours("end_time").alias("end_hour"))
    is_drug = pl.col("event").is_in(DRUGS)
    out = out.with_columns(pl.when(is_drug).then(pl.col("dose_unit")).otherwise(None).alias("dose_unit"),
                           pl.when(is_drug | pl.col("event").is_in(("hospital_admission", "icu_admission")))
                           .then(pl.col("end_hour")).otherwise(None).alias("end_hour"))
    return out.sort("stay_id", "hour", "event")


def main() -> None:
    arch = (ASSETS / "ARCHITECTURE.md").read_text()
    for key in WORLDS:
        d = OUT / key
        if d.exists():
            shutil.rmtree(d)
        for sub in ("model", "model_black", "data"):
            (d / sub).mkdir(parents=True)
        cfg, sd = config(key)
        tensors = {k: list(v.shape) for k, v in sd.items()}
        voc = vocab(key)
        for sub in ("model", "model_black"):
            (d / sub / "tensors.json").write_text(json.dumps(tensors, indent=0) + "\n")
            (d / sub / "vocab.json").write_text(json.dumps({"config": cfg, **voc}, indent=1) + "\n")
            (d / sub / "ARCHITECTURE.md").write_text(arch)
        save_file({k: v.float().contiguous() for k, v in sd.items()}, str(d / "model" / "model.safetensors"))
        for split in ("train", "tuning"):
            f = events(key, split)
            f.write_csv(d / "data" / f"{split}.csv", float_precision=4)
            print(key, split, f.height, "rows", f["stay_id"].n_unique(), "stays", flush=True)
        shutil.copyfile(ASSETS / "DATA_README.md", d / "data" / "README.md")
        print(key, "done:", len(tensors), "tensors,", sum(int(np.prod(s)) for s in tensors.values()), "parameters", flush=True)


if __name__ == "__main__":
    main()
