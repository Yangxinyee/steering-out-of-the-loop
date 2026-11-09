"""API dollars of the agent arm, from each turn's token counts (tokens.jsonl), against a spending cap over all runs ($250
for Claude and Codex together; at the cap the runs stop).

Prices per million tokens at the time of the runs:
  Claude Opus 5.5          input $4, output $20, cache read $0.20, cache write $5 (5-minute) or $8 (1-hour)
  Codex (GPT-6.1 Sol)      input $2.20, cache write $2.75, cache read $0.11, output $11;
                           a request with more than 272,000 input tokens is billed whole at $4.40 / $5.50 / $0.22 / $16.50.
                           A turn's counts may cover several requests, so a turn above 272,000 is priced long: an upper bound.

    python -m agent.budget [agent_runs]        # spent per run and in all, against the cap
"""
import json
import sys
from pathlib import Path

CAP_FILE = "BUDGET.json"            # in the runs folder: {"cap_usd": 250, "stop_share": 0.98, "runs": [names counted]}
LONG = 272_000
CLAUDE = {"input": 4.0, "output": 20.0, "cache_read": 0.20, "cache_write_5m": 5.0, "cache_write_1h": 8.0}
CODEX_SHORT = {"input": 2.20, "cache_write": 2.75, "cache_read": 0.11, "output": 11.0}
CODEX_LONG = {"input": 4.40, "cache_write": 5.50, "cache_read": 0.22, "output": 16.50}


def turn_usd(agent, t):
    inp, cr, cw, out = (t.get(k) or 0 for k in ("input", "cache_read", "cache_write", "output"))
    if agent == "claude":
        w1h = t.get("cache_write_1h")
        w5m = t.get("cache_write_5m")
        if w1h is None and w5m is None:
            w1h, w5m = cw, 0                                        # unknown split: the dearer one
        unc = max(inp - cr - cw, 0)
        return (unc * CLAUDE["input"] + cr * CLAUDE["cache_read"] + (w1h or 0) * CLAUDE["cache_write_1h"]
                + (w5m or 0) * CLAUDE["cache_write_5m"] + out * CLAUDE["output"]) / 1e6
    p = CODEX_LONG if inp > LONG else CODEX_SHORT
    unc = max(inp - cr - cw, 0)
    return (unc * p["input"] + cw * p["cache_write"] + cr * p["cache_read"] + out * p["output"]) / 1e6


def run_usd(d):
    d = Path(d)
    st = json.loads((d / "state.json").read_text())
    f = d / "tokens.jsonl"
    rows = [json.loads(line) for line in open(f)] if f.exists() else []
    return round(sum(turn_usd(st["agent"], r) for r in rows), 2)


def spent(runs_dir):
    """(dollars over the counted runs, cap, stop threshold, {run: dollars}); no BUDGET.json: no cap"""
    runs_dir = Path(runs_dir)
    cf = runs_dir / CAP_FILE
    if not cf.exists():
        return 0.0, None, None, {}
    c = json.loads(cf.read_text())
    per = {n: run_usd(runs_dir / n) for n in c.get("runs", []) if (runs_dir / n / "state.json").exists()}
    total = round(sum(per.values()), 2)
    return total, c["cap_usd"], c["cap_usd"] * c.get("stop_share", 0.98), per


if __name__ == "__main__":
    total, cap, stop, per = spent(sys.argv[1] if len(sys.argv) > 1 else "agent_runs")
    for n, v in per.items():
        print(f"{n:28s} ${v:8.2f}")
    print(f"{'total':28s} ${total:8.2f}" + (f"   cap ${cap}, runs stop at ${stop:.2f}" if cap else "   (no BUDGET.json: no cap)"))
