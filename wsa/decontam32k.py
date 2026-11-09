"""The contamination check against the main baseline (CPU, existing outputs only): the agent's edit minus thinking off at
the full 32,768-token budget, on all test problems and without the near-duplicates of practice problems that wsa.contam
flags; the 8,192-token baseline alongside, as in results/part1_decontam.json. Same paired bootstrap as wsa.analyze.

    python -m wsa.decontam32k        (writes results/decontam32k.json)
"""
import json
from pathlib import Path

import numpy as np

from wsa.analyze import paired

FLAG = {x["id"] for v in json.load(open("results/contam.json")).values() for x in v["flagged"]}


def per(stem, drop):
    return {r["id"]: r["correct"] for r in json.load(open(f"results/runs/{stem}.per_problem.json")) if r["id"] not in drop}


def main():
    out = {}
    for ds in ("math500", "gsm8k"):
        for drop, lab in ((set(), "all"), (FLAG, "clean")):
            e = per(f"edit_authors/think35_{ds}", drop)
            arms = {"thinking_off_32k": per(f"e3/test/nothink35_32k_{ds}", drop),
                    "thinking_off_8k": per(f"base/nothink35_{ds}", drop)}
            r = {"problems": len(e), "edit_acc": float(np.mean([np.mean(v) for v in e.values()]))}
            for k, b in arms.items():
                r[k + "_acc"] = float(np.mean([np.mean(v) for v in b.values()]))
                r["edit_minus_" + k] = paired(e, b)
            out.setdefault(ds, {})[lab] = r
            print(ds, lab, r["problems"], f"edit {r['edit_acc']*100:.1f}", *(
                f"{k} {r[k + '_acc']*100:.1f}: {r['edit_minus_' + k]['gain']*100:+.1f} "
                f"[{r['edit_minus_' + k]['ci95'][0]*100:.1f}, {r['edit_minus_' + k]['ci95'][1]*100:.1f}]" for k in arms))
    Path("results/decontam32k.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
