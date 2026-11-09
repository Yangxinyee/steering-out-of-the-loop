"""Size of the plain model the agents edited: parameters and tensors of each world's seed-0 checkpoint.

    python scripts/model_size.py   -> results/model_size.json (CPU; needs torch and the checkpoints under work/runs/)
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
out = {}
for key in ("e03", "n03"):
    run = f"synth_{key}-m-T1-full-long-s0"
    sd = torch.load(ROOT / "work" / "runs" / run / "model.pt", map_location="cpu", weights_only=False)["model"]
    tensors = {k: v for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}
    out[key] = {"run": run, "n_params": int(sum(v.numel() for v in tensors.values())), "n_tensors": len(tensors)}
(ROOT / "results" / "model_size.json").write_text(json.dumps(out, indent=1) + "\n")
print(out)
