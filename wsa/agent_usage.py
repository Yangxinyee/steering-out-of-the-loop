"""The frontier agents' token use and submitted tensors, for the harness appendix: tokens summed over every turn's
usage record in the harness's run log (relay/runs/<run>/t*.meta.json, not released) and the tensor files each agent
submitted.

    python -m wsa.agent_usage        (writes results/agent_usage.json)
"""
import json
from pathlib import Path

RUNS = {"claude_1": "Claude Opus 5.5", "openai_1": "gpt-6.1-sol"}


def main():
    out = {}
    for run, nm in RUNS.items():
        metas = sorted(Path("relay/runs", run).glob("t*.meta.json"))
        tok = {}
        for f in metas:
            for k, v in json.load(open(f)).get("tokens", {}).items():
                tok[k] = tok.get(k, 0) + (v or 0)
        tensors = sorted(p.name for p in Path("results/runs/agent_eval", run, "submission").glob("*.safetensors"))
        out[nm] = {"turns": len(metas), "input_tokens": tok["input"], "cache_read_tokens": tok["cache_read"],
                   "cache_read_share": tok["cache_read"] / tok["input"], "output_tokens": tok["output"],
                   "tensors": tensors}
    Path("results/agent_usage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
