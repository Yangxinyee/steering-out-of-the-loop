"""Replay an agent's own Python runs, unchanged, in a fresh sandbox on another world's training records.

Used for claude_a: its drug-effect estimate (within-tier regression of later glucose on the insulin rate, deconvolved for
the doses that follow) comes from its Python runs 1-17. Replaying them on the no-effect world asks whether the same
estimator, and so the pathway strength scaled to it, would have invented an effect there. No agent call, no GPU.

    python3 scripts/replay_estimator.py <agent run> <e03|n03> <last python run>
Outputs go to agent_runs/replay/<agent run>_<world>/ (outputs.jsonl, one line per replayed run).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.kernel import Kernel  # noqa: E402

run, world, last = sys.argv[1], sys.argv[2], int(sys.argv[3])
calls = [json.loads(x) for x in open(ROOT / "agent_runs" / run / "tool_calls.jsonl")]
cells = [c for c in calls if c.get("type") == "python" and c.get("run") and c["run"] <= last]
out_dir = ROOT / "agent_runs" / "replay" / f"{run}_{world}"
work = out_dir / "work"
work.mkdir(parents=True, exist_ok=True)
k = Kernel(work, ROOT / "agent_data" / world / "model", ROOT / "agent_data" / world / "data", "white")
with open(out_dir / "outputs.jsonl", "w") as fh:
    for c in cells:
        out, restarted = k.run(c["code"])
        fh.write(json.dumps({"run": c["run"], "restarted": restarted, "output": out}) + "\n")
        fh.flush()
        print(f"run {c['run']}: {len(out)} chars{', restarted: ' + restarted if restarted else ''}", flush=True)
k.kill()
