"""The checkpoint a frontier-agent run submitted (for the evaluation of the agents' edits): its submission/ (the /work/edited tensors at its final answer, agent/harness.py finish()) swapped
into the release exactly as its own forward tool built it (agent.forward.build_checkpoint), so the evaluated weights
are the ones its last passes used. Prints the checkpoint's folder (<out>/<edit hash>/). CPU only.

    python -m wsa.agent_ckpt --submission agent_runs/claude_1/submission --base ckpt/base --out ckpt/agent_claude_1
"""
import argparse
import os
import tempfile
from pathlib import Path

from agent.forward import build_checkpoint, edit_state


def main(a):
    sub = Path(a.submission).resolve()
    assert sub.is_dir() and any(sub.glob("*.safetensors")), f"{sub}: no submitted tensors"
    with tempfile.TemporaryDirectory(dir=a.tmp) as work:
        os.symlink(sub, Path(work) / "edited")
        names, h = edit_state(work)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        ck, h2 = build_checkpoint(Path(a.base), work, Path(a.out))
    assert h == h2
    print(f"{len(names)} edited tensors, edit hash {h}: {ck}", flush=True)
    print(ck)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True, help="a folder of its own: build_checkpoint keeps only the newest build in it")
    ap.add_argument("--tmp", default=None)
    main(ap.parse_args())
