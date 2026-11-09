"""Stay bootstrap of the scripted estimator (scripts/scripted_estimate.py): the training stays resampled with replacement,
the estimator run unchanged, and the insulin x2 shift its lags-1-8 kernel implies on the tuning stays (the target the
script sizes its gain to) recomputed per draw. This is the estimator-level counterpart of the agents' replayed
estimators (scripts/replay_bootstrap.py); the held-out readout's interval reflects only the readout's own noise.

    python3 scripts/scripted_bootstrap.py <e03|n03> <draws> [--workers 8]   -> results/replay_bootstrap/script_<world>.jsonl
Draw -1 is the original records (it must reproduce results/scripted/estimate_<world>.json). CPU, synthetic data only.
"""
import argparse
import contextlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import scripted_estimate as se  # noqa: E402

_TR = _TU = None


def init(world: str) -> None:
    global _TR, _TU
    data = ROOT / "agent_data" / world / "data"
    _TR, _TU = se.panel(data / "train.csv"), se.panel(data / "tuning.csv")


def one(draw: int) -> dict:
    n = len(_TR["stay"])
    if draw < 0:
        p = _TR
    else:
        idx = np.random.default_rng(10_000 + draw).integers(0, n, n)
        p = {k: v[idx] for k, v in _TR.items()}
        p["stay"] = np.arange(n)                                   # each copy its own cluster
    with contextlib.redirect_stdout(io.StringIO()):
        D = se.deviations(p)
        r = se.tsls(p, D, "I", "glucose_finger")
    m, _ = se.implied(_TU, "I", "glucose_finger", se.kernel(r), 2.0)
    return {"run": "script", "draw": draw, "implied_insulin_x2": m, "sum8": float(np.sum(r["psi"][:se.KERNEL_LAGS]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("world", choices=("e03", "n03"))
    ap.add_argument("draws", type=int)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dest = ROOT / "results" / "replay_bootstrap" / f"script_{a.world}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["draw"] for x in open(dest)} if dest.is_file() else set()
    todo = [d for d in range(-1, a.draws) if d not in done]
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.world,)) as ex:
        for r in ex.map(one, todo):
            r["world"] = a.world
            with open(dest, "a") as fh:
                fh.write(json.dumps(r) + "\n")
