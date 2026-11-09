"""The agent's forward-pass tool, outside the sandbox: it checks a request, waits for the GPU, and runs agent/fwd_job.py
as its own process (with the companion code's Python, agent/backend.py; the local card). Nothing the agent wrote
is executed here: the job reads only the edited tensors (white box) or the list of operations (black box) in /work.

GPU rules:
- a job starts only below 85 C and with its peak memory plus 4 GiB free, so concurrency falls by itself when the card is
  hot (>= 90 C) and rises again below 85 C;
- at most GPU_SLOTS jobs at once over all runs (a lock file per slot);
- each job has a deadline (`timeout`), past which it is stopped by its PID; the pass is then not counted.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import backend  # noqa: E402  (the companion paper's code: where it is and which Python runs it)

WM_PY = backend.PYTHON
TYPES = ("factual", "dose_edit", "states")
MODELS = ("original", "edited", "both")
SPLITS = ("train", "tuning")
BUNDLES = ("all", "insulin", "norepinephrine", "drug")
MAX_STAYS = 100
PEAK_MIB, MARGIN_MIB = 2500, 4096
HOT_C = 85


class RequestError(Exception):
    """the request breaks a rule: it fails without being counted"""


def check(req) -> dict:
    if not isinstance(req, dict):
        raise RequestError("the request must be a JSON object")
    extra = set(req) - {"type", "model", "split", "stays", "bundles"}
    if extra:
        raise RequestError(f"unknown fields {sorted(extra)}; allowed: type, model, split, stays, bundles (states only)")
    for k, ok in (("type", TYPES), ("model", MODELS), ("split", SPLITS)):
        if req.get(k) not in ok:
            raise RequestError(f"{k} must be one of {list(ok)}, not {req.get(k)!r}")
    st = req.get("stays")
    if not isinstance(st, list) or not st or not all(isinstance(s, int) and not isinstance(s, bool) for s in st):
        raise RequestError("stays must be a non-empty list of integer stay ids")
    if len(st) > MAX_STAYS:
        raise RequestError(f"at most {MAX_STAYS} stays per pass ({len(st)} given)")
    if len(set(st)) != len(st):
        raise RequestError("stays has repeated ids")
    if "bundles" in req:
        if req["type"] != "states":
            raise RequestError("bundles is for type states only")
        if req["bundles"] not in BUNDLES:
            raise RequestError(f"bundles must be one of {list(BUNDLES)}")
    return req


def gpu() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                             text=True, capture_output=True, timeout=30).stdout.strip().splitlines()[0]
        t, used, total = (float(x) for x in out.split(","))
        return {"temp": t, "free": total - used}
    except Exception:  # noqa: BLE001
        return None


class Forward:
    def __init__(self, run_dir, key, run, mode, slots=3, timeout=1800, log=print, python=WM_PY, seed=0):
        self.d, self.key, self.run_name, self.mode = Path(run_dir), key, run, mode
        self.work = self.d / "work"
        self.slots, self.timeout, self.log, self.python, self.seed = slots, timeout, log, python, seed
        self.vllm = None                                     # the harness's shutdown hook (WSA's vLLM); nothing to stop here
        self.locks = self.d.parent / ".gpu_slots"

    def _slot(self):
        self.locks.mkdir(exist_ok=True)
        waited = 0
        while True:
            for i in range(self.slots):
                f = open(self.locks / f"slot{i}.lock", "w")
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return f
                except OSError:
                    f.close()
            time.sleep(5)
            waited += 5
            if waited % 600 == 0:
                self.log(f"forward: waiting for a GPU slot ({waited // 60} min)")

    def _wait_gpu(self):
        t0, said = time.time(), 0.0
        while True:
            g = gpu()
            if g is None:
                raise RuntimeError("nvidia-smi did not answer")
            if g["temp"] < HOT_C and g["free"] >= PEAK_MIB + MARGIN_MIB:
                if said:
                    self.log(f"forward: GPU ready after {(time.time() - t0) / 60:.1f} min ({g['temp']:.0f} C)")
                return g
            if time.time() - said > 600:
                said = time.time()
                self.log(f"forward: holding the pass: GPU {g['temp']:.0f} C, {g['free']:.0f} MiB free "
                         f"(start below {HOT_C} C with {PEAK_MIB + MARGIN_MIB} MiB free)")
            time.sleep(30)

    def run(self, req, k):
        req = check(req)
        out = self.work / f"fwd_{k}"
        if out.exists():
            raise RequestError(f"/work/fwd_{k} already exists; remove it or let the pass number move on")
        job = {"request": req, "key": self.key, "run": self.run_name, "mode": self.mode, "work": str(self.work),
               "out": str(out), "seed": self.seed}
        jobs = self.d / "requests"
        jobs.mkdir(exist_ok=True)
        jf = jobs / f"fwd_{k}.job.json"
        jf.write_text(json.dumps(job, indent=1))
        slot = self._slot()
        try:
            g = self._wait_gpu()
            env = {k_: v for k_, v in os.environ.items() if k_ not in ("PYTHONPATH",)}
            t0 = time.time()
            with open(jobs / f"fwd_{k}.job.log", "w") as lf:
                p = subprocess.Popen([self.python, str(HERE / "fwd_job.py"), str(jf)], stdout=lf, stderr=subprocess.STDOUT,
                                     cwd=str(REPO), env=env)
                self.log(f"forward {k}: job PID {p.pid} started ({req['type']}, {req['model']}, {len(req['stays'])} stays; "
                         f"GPU {g['temp']:.0f} C, {g['free']:.0f} MiB free)")
                try:
                    p.wait(timeout=self.timeout)
                except subprocess.TimeoutExpired:
                    p.kill()                                    # by its PID
                    p.wait()
                    raise RequestError(f"the pass took longer than {self.timeout} s and was stopped") from None
            dt = time.time() - t0
        finally:
            fcntl.flock(slot, fcntl.LOCK_UN)
            slot.close()
        sf = out / "summary.json"
        if not sf.exists():
            raise RuntimeError(f"the forward job ended without a summary (exit {p.returncode})")
        s = json.loads(sf.read_text())
        if not s.get("ok"):
            if s.get("kind") == "internal":
                self.log(f"forward {k}: internal error: {s.get('error')}")
                raise RuntimeError("the forward tool failed on its side (" + s["error"][:300] + ")")
            raise RequestError(s["error"])
        self.log(f"forward {k}: done in {dt:.0f} s, GPU peak {s.get('gpu_peak_mib')} MiB")
        summary = {"type": s["type"], "model": req["model"], "split": s["split"], "n_stays": s["n_stays"],
                   "results": s["results"], "seconds": s["seconds"]}
        edited = s.get("edited_tensors") or []
        return summary, edited, s.get("edit_hash", "none")
