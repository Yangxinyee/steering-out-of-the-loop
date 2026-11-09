"""The one place where this code reaches the companion paper's code: the ICU simulator, the event-stream world model and
its data pipeline, the teacher-forced and rollout evaluations, and the evaluation of the control corrections. That code is
released with the companion paper; this module only knows where it is and what to call, from a JSON file:

    world_model.json at the repository root (copy world_model.example.json), or the file named by $WORLD_MODEL_CONFIG

    {"home":   the companion code's directory,
     "python": the Python that has its dependencies (torch, polars, ...); used to launch forward-pass jobs,
     "paths":  directories under "home" to put on sys.path,
     "env":    {"work_root" | "cohort" | "mhat_file": the companion code's environment variable for each setting},
     "names":  {neutral name: "module" or "module:attribute"}}

Usage: set the companion code's settings before its modules are first used, then import what is needed by its neutral
name; nothing is imported until it is asked for.

    import backend
    backend.configure(work_root=ROOT / "work", cohort="synth_e03")
    from backend import rollout, WindowSpec, WorldModel

Neutral names used here: sim (the ICU simulator), dataset (the bundle store and loaders; also BundleStore, WindowSpec,
make_loader), observations (HideObservations), ModelConfig, Vocab, WorldModel, load_actions, rollout (held-out dose-edit
rollouts), evaluation (held-out cohort and cuts), teacher_forced (per-head held-out losses), baselines (readout summaries
of the control corrections).
"""
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    f = Path(os.environ.get("WORLD_MODEL_CONFIG", ROOT / "world_model.json"))
    if not f.is_file():
        raise SystemExit(f"{f} not found: copy world_model.example.json to world_model.json and fill in where the "
                         "companion paper's code lives (or set WORLD_MODEL_CONFIG)")
    return json.loads(f.read_text())


CFG = _load()
HOME = Path(CFG["home"])
PYTHON = CFG.get("python") or sys.executable
for _p in CFG.get("paths", []):
    if str(HOME / _p) not in sys.path:
        sys.path.insert(0, str(HOME / _p))


def configure(**settings) -> None:
    """set the companion code's settings (work_root, cohort, mhat_file) before its modules are first imported"""
    for k, v in settings.items():
        os.environ[CFG["env"][k]] = str(v)


def setdefault(**settings) -> None:
    """as configure(), but keep a value already set in the environment"""
    for k, v in settings.items():
        os.environ.setdefault(CFG["env"][k], str(v))


def env_name(setting: str) -> str:
    """the environment variable the companion code reads for a setting"""
    return CFG["env"][setting]


def __getattr__(name: str):
    spec = CFG["names"].get(name)
    if spec is None:
        raise AttributeError(f"backend has no '{name}' (add it to the config's \"names\")")
    module, _, attr = spec.partition(":")
    obj = importlib.import_module(module)
    obj = getattr(obj, attr) if attr else obj
    globals()[name] = obj
    return obj
