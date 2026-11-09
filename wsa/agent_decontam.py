"""The frontier agents' edits without the MATH-500 near-duplicates of practice problems (CPU, existing
outputs): accuracy and paired differences against the base model, thinking off at the full budget and the published
edit, on all 500 problems and on the 389 that wsa.contam does not flag. Same paired bootstrap as wsa.analyze.

    python -m wsa.agent_decontam        (writes results/agent_decontam.json)
"""
import json
from pathlib import Path

import numpy as np

from wsa.analyze import paired

FLAG = {x["id"] for v in json.load(open("results/contam.json")).values() for x in v["flagged"]}
ARMS = {"gpt-6.1-sol": "agent_eval/openai_1/think35_math500", "Claude Opus 5.5": "agent_eval/claude_1/think35_math500"}
REFS = {"base": "base/think35_math500", "thinking off, full budget": "e3/test/nothink35_32k_math500",
        "published edit": "edit_authors/think35_math500"}


def per(stem, drop):
    return {r["id"]: r["correct"] for r in json.load(open(f"results/runs/{stem}.per_problem.json")) if r["id"] not in drop}


def main():
    out = {}
    for lab, drop in (("all", set()), ("clean", FLAG)):
        R = {}
        for a, st in ARMS.items():
            e = per(st, drop)
            R[a] = {"problems": len(e), "acc": float(np.mean([np.mean(v) for v in e.values()]))}
            for rn, rs in REFS.items():
                R[a]["minus " + rn] = paired(e, per(rs, drop))
            print(lab, a, R[a]["problems"], f"acc {R[a]['acc']*100:.1f}", *(
                f"- {rn}: {R[a]['minus ' + rn]['gain']*100:+.1f} [{R[a]['minus ' + rn]['ci95'][0]*100:.1f}, {R[a]['minus ' + rn]['ci95'][1]*100:.1f}]"
                for rn in REFS))
        out[lab] = R
    Path("results/agent_decontam.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
