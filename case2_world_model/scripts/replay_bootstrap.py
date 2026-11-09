"""Stay bootstrap of an agent's estimator replayed on a world's training records: the agent's own Python runs, unchanged,
in a fresh sandbox (as scripts/replay_estimator.py), on copies of train.csv resampled by stay with replacement. Each
draw gives the insulin x2 shift the estimator implies, so its 95% interval can be set against the no-effect band.

    python3 scripts/replay_bootstrap.py <claude_a|openai_a> <e03|n03> <first draw> <last draw> [--workers 4]
draw -1 is the original records (it must reproduce the replay's point estimate). One line per draw is appended to
results/replay_bootstrap/<run>_<world>.jsonl. CPU only; synthetic data only.

The implied shift is computed as for the point estimate (scripts/paper_numbers.py):
  claude_a  its finger-stick effects by lag (Python run 16, after removing the feedback of later doses), as the ratio of
            the cumulative effect averaged over the readout's twelve hours to the main world's, times Claude's own
            estimate of the insulin x2 readout, to which it matched its circuit
  openai_a  the sustained insulin x2 shift its discontinuity fit and persistence imply on the training stays (run 24)
"""
import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LAST = {"claude_a": 17, "openai_a": 24}
BOOT = ROOT / "work" / "replay_boot"


def resampled(run: str, world: str, draw: int) -> Path:
    """a data folder whose train.csv holds 30,000 stays drawn with replacement (each copy a new stay id, rows in the
    original order within a stay); tuning.csv and README.md are the originals"""
    import polars as pl
    src = ROOT / "agent_data" / world / "data"
    if draw < 0:
        return src
    d = BOOT / f"data_{run}_{world}_{draw}"                    # the same draw is the same resample for every agent
    if (d / "train.csv").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    tr = pl.read_csv(src / "train.csv", infer_schema_length=0).with_row_index("_row")    # every column as text
    ids = tr["stay_id"].unique(maintain_order=True).to_numpy()
    pick = np.random.default_rng(10_000 + draw).choice(ids, size=len(ids), replace=True)
    base = int(ids.astype(np.int64).min())        # stay ids run from base to base + 29,999; one agent's code relies on it
    m = pl.DataFrame({"stay_id": pick, "new_id": (base + np.arange(len(pick))).astype(str)})
    out = tr.join(m, on="stay_id", how="inner").sort(["new_id", "_row"])
    out = out.with_columns(pl.col("new_id").alias("stay_id")).drop("new_id", "_row")
    out.write_csv(d / "train.csv.part")
    (d / "train.csv.part").rename(d / "train.csv")
    for f in ("tuning.csv", "README.md"):
        shutil.copy2(src / f, d / f)
    return d


def implied(run: str, outputs: dict) -> float:
    if run == "claude_a":
        def psi(text):
            return np.array([float(v) for v in re.search(r"psi finger\s*\[([^\]]*)\]", text).group(1).split()])
        main = [json.loads(x) for x in open(ROOT / "agent_runs" / "replay" / "claude_a_e03" / "outputs.jsonl")]
        a = psi(next(o["output"] for o in main if o["run"] == 16))
        b = psi(outputs[16])
        fa = (ROOT / "agent_runs" / "claude_a" / "final_answer.md").read_text()
        own = float(re.search(r"Insulin x2[^|]*\|[^|]*\|[^|]*\|\s*([−+\-0-9.]+)", fa).group(1).replace("−", "-"))
        return float(own * np.cumsum(b).mean() / np.cumsum(a).mean())
    tg = re.search(r"train / first 100 tuning: \(([-0-9.e]+), \d+\)", outputs[24])
    return float(tg.group(1))


def one(args) -> dict:
    run, world, draw = args
    from agent.kernel import Kernel
    data = resampled(run, world, draw)
    work = BOOT / f"work_{run}_{world}_{draw}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
    cells = [c for c in calls if c.get("type") == "python" and c.get("run") and c["run"] <= LAST[run]]
    k = Kernel(work, ROOT / "agent_data" / world / "model", data, "white")
    outputs, restarts = {}, []
    try:
        for c in cells:
            out, restarted = k.run(c["code"])
            outputs[c["run"]] = out
            if restarted:
                restarts.append(c["run"])
    finally:
        k.kill()
    try:
        v = implied(run, outputs)
    except Exception as e:  # noqa: BLE001
        v, err = None, f"{type(e).__name__}: {e}"
    else:
        err = None
    shutil.rmtree(work, ignore_errors=True)
    if draw >= 0:
        shutil.rmtree(data, ignore_errors=True)
    return {"run": run, "world": world, "draw": draw, "implied_insulin_x2": v, "restarts": restarts, "error": err}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=sorted(LAST))
    ap.add_argument("world", choices=("e03", "n03"))
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dest = ROOT / "results" / "replay_bootstrap" / f"{a.run}_{a.world}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["draw"] for x in open(dest)} if dest.is_file() else set()
    todo = [(a.run, a.world, d) for d in range(a.first, a.last + 1) if d not in done]
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(one, todo):
            with open(dest, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(json.dumps(r), flush=True)
