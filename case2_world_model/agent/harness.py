"""One agent run, adapted from case 1's agent harness. The agent answers through a responder (agent/responder.py): in the
paper's runs, a transport (not included) to a separate machine where the agent's CLI answered; for dry runs, a scripted
fake (--responder fake).

Every turn the harness sends one message. The agent replies with an explanation and exactly one ```python or ```forward
block, or with a final answer (no block, starting with FINAL ANSWER), and the harness runs the block:
  python    in the sandboxed kernel (agent/kernel.py), at most 300 runs; the output, ASCII only and cut at 10,000
            characters, is the next message ("OUTPUT of Python run k of at most 300:")
  forward   through the forward tool (agent/forward.py, the local GPU), at most 50 passes; a failed request is not counted
A fresh conversation starts every `handoff` replies, with the task, the counts, the current edits, the files in /work,
the last 8,000 characters of /work/notes.md and the last result. Where a reply has no block, two blocks, or a block
without an explanation, or the runs are used up, the harness says so in one bracketed sentence.

The task message is the same in both worlds (blind); the arms differ only in the kernel's API (agent/assets/arm_*.txt).
The run's world, model and arm are kept in state.json on this machine and never appear in a message: agent paths, user,
host and the world's names are replaced before anything leaves (scrub).

Memory: before a turn goes out, the harness waits while the user service's memory pressure is above 30% and sends the
turn once it is below 10%; a running kernel is stopped above 45% (agent/kernel.py). A file STOP in the run folder stops
the run between turns (before the next turn goes out, or while waiting for the agent's answer); `run` again continues.

    python -m agent.harness init <run dir> --agent claude|openai --world e03|n03 --arm white|black [--handoff 40]
    python -m agent.harness run <run dir> [--responder fake]    (a real run plugs in a responder: agent/responder.py)
    touch <run dir>/STOP
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
ASSETS = HERE / "assets"
DATA = REPO / "agent_data"
OUT_CHARS, NOTES_CHARS, MIN_EXPLAIN, EXPLAIN_SENDBACKS = 10_000, 8_000, 100, 2
BLOCK = re.compile(r"```(python|forward)[^\n]*\n(.*?)```", re.S)
FINAL = re.compile(r"^\W*FINAL ANSWER", re.I)
RESTART_NOTE = "[note: the harness was restarted, so your kernel restarted too and its variables are gone]"
REFUSED_NOTE = ASSETS / "refused_note.txt"
WORLD_NAMES = ("synth_e03_td100", "synth_e03", "synth_n03", "s+1_b1_e0.3_td100", "s+1_b1_e0.3", "s+0_b1_e0.3", "s-2_b1_e0.3")
RUN_OF = {"e03": "synth_e03-m-T1-full-long-s0", "n03": "synth_n03-m-T1-full-long-s0"}


def scrub(text):
    """no host detail leaves the workstation (its paths, user and host name), nor the world's names"""
    for s, r in ((str(DATA), "<data>"), (str(REPO), "<repo>"), (str(Path.home()), "~"), (socket.gethostname(), "<host>"),
                 (os.environ.get("USER", "\0"), "<user>")):
        if s and len(s) > 2:
            text = text.replace(s, r)
    for w in WORLD_NAMES:
        text = text.replace(w, "<world>")
    return re.sub(r"arn:aws[\w-]*:\S*?(?=[\s,]|$)", "<arn>", text)


def ascii_only(s):
    return "".join(c if ord(c) < 128 else "?" for c in s)


def task_message(arm, handoff):
    return ((ASSETS / "message.txt").read_text().replace("{ARM}", (ASSETS / f"arm_{arm}.txt").read_text().rstrip("\n"))
            .replace("{HANDOFF}", str(handoff)))


def edit_state(work, arm):
    """(names of the edited tensors, a short hash of the edit), the way agent/fwd_job.py hashes it"""
    work = Path(work)
    if arm == "white":
        files = sorted((work / "edited").glob("*.safetensors")) if (work / "edited").is_dir() else []
        names = [f.name[: -len(".safetensors")] for f in files]
    else:
        ops = json.loads((work / "ops.json").read_text()) if (work / "ops.json").exists() else []
        files = ([work / "ops.json"] + sorted({work / o["vector"] for o in ops if o.get("vector")})) if ops else []
        names = [f"{i + 1}. {o['op']} {o['name']}" for i, o in enumerate(ops)]
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return names, (h.hexdigest()[:10] if files else "none")


class Run:
    def __init__(self, d):
        self.d = Path(d)
        self.work = self.d / "work"
        self.s = json.loads((self.d / "state.json").read_text())

    def save(self):
        tmp = self.d / "state.json.tmp"
        tmp.write_text(json.dumps(self.s, indent=1))
        tmp.replace(self.d / "state.json")

    def log(self, msg):
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {msg}"
        print(line, flush=True)
        with open(self.d / "driver.log", "a") as f:
            f.write(line + "\n")

    def tool_call(self, rec):
        with open(self.d / "tool_calls.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def init(a):
    d = Path(a.run_dir)
    assert not (d / "state.json").exists(), f"{d} already has a run"
    (d / "work").mkdir(parents=True)
    (d / "turns").mkdir()
    s = {"name": d.name, "agent": a.agent, "world": a.world, "arm": a.arm, "model_run": a.model_run or RUN_OF[a.world],
         "budget": a.budget, "max_runs": a.max_runs, "handoff": a.handoff, "handoff_tokens": a.handoff_tokens,
         "turn": 0, "runs": 0, "passes": 0, "requests": 0, "thread_replies": 0, "new_thread": True, "done": False,
         "nudges": 0, "handoffs": 0, "flags": 0, "refusals": 0, "unexplained": 0, "unexplained_streak": 0,
         "kernel_started": False, "next": task_message(a.arm, a.handoff), "last_result": None,
         "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (d / "state.json").write_text(json.dumps(s, indent=1))
    print(f"{d}: run initialized ({a.agent}, world {a.world}, {a.arm} box, {a.budget} passes, {a.max_runs} Python runs, "
          f"handoff every {a.handoff} replies or past {a.handoff_tokens:,} input tokens)")


def handoff_message(r, kernel_alive):
    s = r.s
    notes = ""
    if (r.work / "notes.md").exists():
        notes = (r.work / "notes.md").read_text(errors="replace")[-NOTES_CHARS:]
    edited, _ = edit_state(r.work, s["arm"])
    files = sorted(os.listdir(r.work))
    kern = ("Your kernel is still running with its variables." if kernel_alive
            else "Your kernel was restarted, so its variables are gone.")
    what = "Tensors currently edited" if s["arm"] == "white" else "Operations so far"
    return (task_message(s["arm"], s["handoff"]) + f"\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {s['turn']} "
            f"replies happened before. Forward passes used: {s['passes']} of {s['budget']}; Python runs used: {s['runs']} of "
            f"{s['max_runs']}. {kern} {what}: {edited}. Files in /work: {files}.\n"
            f"YOUR NOTEBOOK (/work/notes.md, last 8,000 characters):\n```\n{notes}\n```\nContinue from here.\n\n"
            f"Last result:\n{s['last_result']}")


def forward_message(r, summary, edited, h, model, k):
    s = r.s
    left = s["budget"] - s["passes"]
    msg = (f"Forward pass {s['passes']} of {s['budget']} done ({summary['type']}); files in /work/fwd_{k}/. Summary:\n```\n"
           f"{json.dumps(summary, indent=1)}\n```\nForward passes left: {left}.")
    if model in ("edited", "both"):
        msg += f"\nEdits used in this pass: {', '.join(edited) if edited else 'none'} (edit hash {h})."
    if left == 0:
        msg += "\nAll forward passes are used. Make your final choice of edits (Python is still available), then reply FINAL ANSWER without a code block."
    return msg


def finish(r, answer):
    edited, h = edit_state(r.work, r.s["arm"])
    sub = r.d / "submission"
    if sub.exists():
        shutil.rmtree(sub)
    sub.mkdir()
    if r.s["arm"] == "white" and (r.work / "edited").is_dir():
        shutil.copytree(r.work / "edited", sub / "edited")
    if r.s["arm"] == "black" and (r.work / "ops.json").exists():
        shutil.copyfile(r.work / "ops.json", sub / "ops.json")
        if (r.work / "vectors").is_dir():
            shutil.copytree(r.work / "vectors", sub / "vectors")
    (r.d / "final_answer.md").write_text(answer)
    r.s.update(done=True, submitted=edited, edit_hash=h)
    from agent.budget import run_usd
    r.s["usd"] = run_usd(r.d)
    r.log(f"final answer: {r.s['runs']} python runs, {r.s['passes']} forward passes, {r.s['nudges']} nudges, "
          f"{r.s['handoffs']} handoffs, {len(edited)} edits (edit hash {h}); API ${r.s['usd']:.2f}")


def wait_for_memory(r, t, psi, pause_at, resume_at):
    """hold turn t while the memory pressure is high (between turns only)"""
    from agent.kernel import memory_pressure
    pr = memory_pressure(psi)
    if pr is None or pr["level"] <= pause_at:
        return
    r.log(f"turn {t}: memory pressure {pr['level']:.1f}% (avg10) is over {pause_at:.0f}%; holding the turn until it "
          f"falls below {resume_at:.0f}%")
    t0 = time.time()
    while pr is not None and pr["level"] >= resume_at:
        time.sleep(10)
        pr = memory_pressure(psi)
    dt = time.time() - t0
    r.s["pressure_pauses"] = r.s.get("pressure_pauses", 0) + 1
    r.s["pressure_pause_s"] = round(r.s.get("pressure_pause_s", 0) + dt)
    r.log(f"turn {t}: memory pressure {pr['level'] if pr else float('nan'):.1f}%; the turn goes out after {dt / 60:.1f} min")


def step(r, kernel, fwd, responder, mem=None):
    """one turn: send the pending message, run what comes back, store the next message"""
    s = r.s
    t = s["turn"] + 1
    # keep every request under about 250,000 input tokens (the short-context price); a fresh
    # conversation starts after `handoff` replies, or once the last turn's input passed `handoff_tokens`
    long_ = (s.get("last_input") or 0) >= s.get("handoff_tokens", 10**12)
    if not s["new_thread"] and (s["thread_replies"] >= s["handoff"] or long_):
        s["new_thread"], s["handoffs"] = True, s["handoffs"] + 1
        s["next"] = handoff_message(r, kernel.alive() and not s.get("kernel_reset"))
    thread = "new" if s["new_thread"] else "resume"
    if mem:
        wait_for_memory(r, t, **mem)
    prompt = scrub(s["next"])
    td = r.d / "turns" / f"t{t:03d}"
    td.mkdir(parents=True, exist_ok=True)
    (td / "prompt.txt").write_text(prompt)
    r.log(f"turn {t}: {thread} ({len(prompt):,} chars)")
    r.save()                                                  # a stop while the answer is awaited resumes from here
    answer, meta = responder.ask(s["name"], t, prompt, thread)
    (td / "answer.md").write_text(answer)
    (td / "meta.json").write_text(json.dumps(meta, indent=1))
    tk = meta.get("tokens") or {}
    with open(r.d / "tokens.jsonl", "a") as f:                # every turn's token use (agent/usage.py sums it)
        f.write(json.dumps({"turn": t, "thread": thread, "wall_s": meta.get("wall_s"), "attempts": meta.get("attempts"),
                            "refused": bool(meta.get("refused")), **tk}) + "\n")
    tot = s.setdefault("tokens", {})
    for key in ("input", "cache_read", "cache_write", "output", "reasoning_output"):
        tot[key] = tot.get(key, 0) + (tk.get(key) or 0)
    s["agent_wall_s"] = round(s.get("agent_wall_s", 0) + (meta.get("wall_s") or 0), 1)
    s["last_input"] = tk.get("input") or 0
    from agent.budget import run_usd
    s["usd"] = run_usd(r.d)
    if tk:
        r.log(f"turn {t}: tokens in {tk.get('input', 0):,} (cache read {tk.get('cache_read', 0):,}, write "
              f"{tk.get('cache_write', 0):,}), out {tk.get('output', 0):,}; run so far in {tot['input']:,}, out {tot['output']:,}")
    s["turn"], s["new_thread"] = t, False
    s["thread_replies"] = 1 if thread == "new" else s["thread_replies"] + 1
    if meta.get("tool_free") is False:
        s["flags"] += 1
        r.log(f"turn {t}: the agent used its own tools ({meta.get('tool_items')}): flagged")
    if meta.get("refused"):
        s["refusals"] += 1
        s["next"] = REFUSED_NOTE.read_text().strip()
        r.log(f"turn {t}: refused by the content filter ({s['refusals']} so far)")
        return
    blocks = BLOCK.findall(answer)
    if not blocks:
        if FINAL.match(answer.strip()) or s["passes"] >= s["budget"]:
            finish(r, answer)
            return
        s["nudges"] += 1
        s["nudge_streak"] = s.get("nudge_streak", 0) + 1
        left = s["budget"] - s["passes"]
        s["next"] = (f"[Your reply has no ```python or ```forward block and does not start with FINAL ANSWER ({left} forward "
                     "passes left). Reply with an explanation and exactly one block, or with your final answer.]")
        r.log(f"turn {t}: no block; nudge {s['nudges']}")
        return
    s["nudge_streak"] = 0
    if len(blocks) > 1:
        s["next"] = (f"[Your reply has {len(blocks)} blocks, so none was run. Reply with an explanation and exactly "
                     "one ```python or ```forward block.]")
        r.log(f"turn {t}: {len(blocks)} blocks; none run")
        return
    kind, body = blocks[0]
    before = answer[:answer.find("```")].strip()
    if len(before) < MIN_EXPLAIN and s["unexplained_streak"] < EXPLAIN_SENDBACKS:
        s["unexplained"] += 1
        s["unexplained_streak"] += 1
        s["next"] = ("[Your reply has no explanation before its block, so the block was not run. Explain in plain "
                     "English what the last result showed, what you conclude from it, what you will do now and why, "
                     "and what outcome would confirm or refute your current hypothesis; then send the block again.]")
        r.log(f"turn {t}: {kind} block without an explanation; sent back ({s['unexplained_streak']} in a row)")
        return
    s["unexplained_streak"] = 0
    if kind == "python":
        if s["runs"] >= s["max_runs"]:
            s["next"] = f"[You have used all {s['max_runs']} Python runs, so the block was not run.]"
            r.log(f"turn {t}: python past the limit; not run")
            return
        t0 = time.time()
        out, restarted = kernel.run(body)
        s["runs"] += 1
        pre = ""
        if restarted == "start" and s["kernel_started"]:
            pre = RESTART_NOTE + "\n"
        elif restarted in ("time limit", "memory limit or crash", "memory pressure"):
            out += " The kernel was restarted, so its variables are gone."
        if restarted == "memory pressure":
            s["pressure_kills"] = s.get("pressure_kills", 0) + 1
            r.log(f"turn {t}: memory pressure {kernel.last_stats.get('pressure_kill')} while the code ran: the kernel "
                  f"was stopped ({s['pressure_kills']} so far)")
        s["kernel_started"] = True
        s["kernel_reset"] = restarted in ("time limit", "memory limit or crash", "memory pressure")
        st = kernel.last_stats
        s["kernel_peak_gb"] = max(s.get("kernel_peak_gb", 0), st.get("mem_peak_gb") or 0, st.get("python_peak_gb") or 0)
        out = ascii_only(out)
        if len(out) > OUT_CHARS:
            out = out[:OUT_CHARS] + "\n[output cut at 10,000 characters]"
        s["next"] = pre + f"OUTPUT of Python run {s['runs']} of at most {s['max_runs']}:\n```\n{out}\n```"
        r.tool_call({"turn": t, "type": "python", "run": s["runs"], "code": body, "output": out,
                     "s": round(time.time() - t0, 1), "restarted": restarted, **st})
        r.log(f"turn {t}: python; runs {s['runs']}, passes {s['passes']}/{s['budget']}; kernel memory peak "
              f"{st.get('mem_peak_gb', '?')} GB (scope), {st.get('python_peak_gb', '?')} GB (process)")
        return
    if s["passes"] >= s["budget"]:
        s["next"] = "[All forward passes are used, so the request was not run. Make your final choice of edits, then reply FINAL ANSWER without a code block.]"
        r.log(f"turn {t}: forward past the budget; not run")
        return
    s["requests"] += 1
    k = s["requests"]
    (r.d / "requests").mkdir(exist_ok=True)
    (r.d / "requests" / f"fwd_{k}.json").write_text(body)
    try:
        req = json.loads(body)
        summary, edited, h = fwd.run(req, k)
    except Exception as e:  # noqa: BLE001  any failure (a broken rule, bad JSON, the tool itself): not counted
        err = f"{type(e).__name__}: {e}"
        s["next"] = f"The forward pass failed and was not counted: {err}"
        r.tool_call({"turn": t, "type": "forward", "request": k, "ok": False, "error": err})
        r.log(f"turn {t}: forward request {k} failed ({err[:200]}); passes {s['passes']}/{s['budget']}")
        s["last_result"] = s["next"]
        return
    s["passes"] += 1
    s["next"] = forward_message(r, summary, edited, h, req.get("model"), k)
    (r.d / "requests" / f"fwd_{k}.done.json").write_text(json.dumps(summary, indent=1))
    r.tool_call({"turn": t, "type": "forward", "request": k, "ok": True, "pass": s["passes"], "edit_hash": h, "summary": summary})
    r.log(f"turn {t}: forward; runs {s['runs']}, passes {s['passes']}/{s['budget']}")


def stopped(r, when):
    (r.d / "STOP").unlink(missing_ok=True)
    r.s.setdefault("stops", []).append({"after_turn": r.s["turn"], "when": when, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    r.save()
    r.log(f"stopped by the STOP file after turn {r.s['turn']}, {when}; passes {r.s['passes']}/{r.s['budget']}, "
          f"runs {r.s['runs']}; state saved")


def run(a):
    from agent.forward import Forward
    from agent.kernel import Kernel
    r = Run(a.run_dir)
    s = r.s
    model_dir = DATA / s["world"] / ("model" if s["arm"] == "white" else "model_black")
    kernel = Kernel(r.work, model_dir, DATA / s["world"] / "data", s["arm"], mem=a.mem, cpus=a.cpus, timeout=a.timeout,
                    psi=a.psi, kill_at=a.kill_at)
    mem = {"psi": a.psi, "pause_at": a.pause_at, "resume_at": a.resume_at}
    fwd = Forward(r.d, s["world"], s["model_run"], s["arm"], slots=a.gpu_slots, timeout=a.fwd_timeout, log=r.log)
    from agent.dryrun.fake_agent import LocalResponder      # the scripted dry run; a real run plugs in a responder
    responder = LocalResponder(r.d)                          # with the interface of agent/responder.py
    from agent.responder import Stopped                     # not defined here: this module runs as __main__
    if s["turn"] or s.get("stops"):
        s.setdefault("restarts", []).append({"after_turn": s["turn"], "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        r.log(f"harness restarted after turn {s['turn']} (passes {s['passes']}/{s['budget']}, runs {s['runs']}); "
              "the kernel starts afresh")
        r.save()
    from agent.budget import spent
    try:
        while not r.s["done"]:
            if (r.d / "STOP").exists():
                stopped(r, "before the next turn went out")
                break
            usd, cap, stop_at, _ = spent(r.d.parent)              # the API spending cap over all runs; at the cap, stop
            if cap is not None and usd >= stop_at:
                r.s.setdefault("budget_stops", []).append({"after_turn": r.s["turn"], "usd_all_runs": usd, "cap": cap,
                                                           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
                r.log(f"API budget: ${usd:.2f} spent over all runs, cap ${cap} (stop at ${stop_at:.2f}): stopping after turn "
                      f"{r.s['turn']}; state saved, `run` again continues once the cap is raised")
                break
            try:
                step(r, kernel, fwd, responder, mem)
            except Stopped:
                stopped(r, f"while waiting for the answer to turn {r.s['turn'] + 1} (it was sent; a resumed run waits for it)")
                break
            r.s["last_result"] = r.s["next"] if not r.s["done"] else r.s["last_result"]
            r.save()
            if a.max_turns and r.s["turn"] >= a.max_turns:
                r.log(f"stopping after turn {r.s['turn']} (--max-turns)")
                break
            if r.s.get("nudge_streak", 0) >= a.max_nudge_streak:   # a guard against an endless, billed loop
                r.log(f"stopping after turn {r.s['turn']}: {r.s['nudge_streak']} replies in a row without a block")
                break
    finally:
        kernel.kill()
        r.save()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("run_dir")
    p.add_argument("--agent", choices=("claude", "openai"), required=True)
    p.add_argument("--world", choices=("e03", "n03"), required=True)
    p.add_argument("--arm", choices=("white", "black"), required=True)
    p.add_argument("--model-run", default=None, help="the plain model (default: seed 0 of the world)")
    p.add_argument("--budget", type=int, default=50)
    p.add_argument("--max-runs", type=int, default=300)
    p.add_argument("--handoff", type=int, default=40)
    p.add_argument("--handoff-tokens", type=int, default=180_000,
                   help="a fresh conversation once a turn's input passes this (the next request then stays under ~250,000)")
    p = sub.add_parser("run")
    p.add_argument("run_dir")
    p.add_argument("--responder", choices=("fake",), default="fake")
    p.add_argument("--mem", default="32G", help="the kernel's memory cap")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--timeout", type=int, default=900, help="seconds per Python run")
    p.add_argument("--gpu-slots", type=int, default=3, help="forward jobs on the card at once, over all runs")
    p.add_argument("--fwd-timeout", type=int, default=1800, help="seconds per forward job (then stopped by its PID)")
    p.add_argument("--max-turns", type=int, default=0, help="stop after this turn (testing)")
    p.add_argument("--max-nudge-streak", type=int, default=10, help="stop after this many replies in a row without a block")
    p.add_argument("--psi", default=f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/memory.pressure")
    p.add_argument("--pause-at", type=float, default=30.0, help="hold the next turn above this pressure (%)")
    p.add_argument("--resume-at", type=float, default=10.0, help="send it again below this pressure (%)")
    p.add_argument("--kill-at", type=float, default=45.0, help="stop a running kernel above this pressure (%)")
    a = ap.parse_args()
    init(a) if a.cmd == "init" else run(a)
