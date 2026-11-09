"""A scripted stand-in for the agent, for dry runs of the whole loop without any model call (adapted from case 1's kit):
the harness, the sandboxed kernel and the forward tool on the GPU (python -m agent.harness run <dir> --responder fake).

The white-box script touches every path once:
- sandbox checks (no network, no home, no data disk, no GPU);
- a block without an explanation (sent back);
- an edit;
- all three request types, with "both";
- reading the result files;
- a content-filter refusal (the marker below makes the stand-in refuse);
- a request with an unknown stay and one with too many stays (both fail, not counted);
- two blocks in one reply;
- a Python run past the time limit;
- a reply with no block (nudged);
- a handoff (run it with a small --handoff);
- the final answer.

The black-box script checks that no weights are mounted, takes states, builds a rank-1 operation and a scale from them,
runs the edited model, undoes one operation and submits.
"""
import json
import os
import time
from pathlib import Path

REFUSE_MARKER = "[[dryrun-refuse]]"
E = ("The last result is what I expected at this stage of the dry run. I conclude that the tool path works as "
     "described. Next I take the following step to exercise the next part of the harness, and a clean result confirms it.")
PY = "```python\n{}\n```"
FW = "```forward\n{}\n```"
TU = [2020000000, 2020000001, 2020000002, 2020000003, 2020000004]
TR = [1010000000, 1010000001, 1010000002, 1010000003, 1010000004]
SANDBOX = """# hypothesis: the sandbox has only /model, /data, /work and no network, home or GPU
import os, socket, glob, pandas as pd
print(sorted(os.listdir('/')))
print('home', os.path.exists('/home'), 'media', os.path.exists('/media'), 'gpu', glob.glob('/dev/nvidia*'))
try:
    socket.create_connection(('1.1.1.1', 80), timeout=3); print('network: OPEN')
except OSError as e:
    print('network: blocked', type(e).__name__)
tr = pd.read_csv('/data/train.csv'); tu = pd.read_csv('/data/tuning.csv')
open('/work/notes.md', 'w').write('# dry run\\n')
print(len(NAMES), tr.shape, tu.shape, tr.stay_id.nunique(), sorted(os.listdir('/model')))"""

WHITE = [
    E + "\n\n" + PY.format(SANDBOX),
    PY.format("print('a block without an explanation')"),
    E + "\n\n" + PY.format("# hypothesis: an edit of the action encoder saves and is listed\n"
                           "W = load_original('encode_action.mlp2.2.weight')\nprint(save_edit('encode_action.mlp2.2.weight', 1.2 * W))\nedited()"),
    E + "\n\n" + FW.format(json.dumps({"type": "factual", "model": "both", "split": "tuning", "stays": TU})),
    E + "\n\n" + FW.format(json.dumps({"type": "dose_edit", "model": "both", "split": "train", "stays": TR})),
    E + "\n\n" + FW.format(json.dumps({"type": "states", "model": "edited", "split": "tuning", "stays": TU[:3], "bundles": "insulin"})),
    E + "\n\n" + PY.format(
        "# hypothesis: the files of the three passes are readable\nimport pandas as pd, numpy as np\n"
        "f = pd.read_csv('/work/fwd_1/factual_edited.csv'); d = pd.read_csv('/work/fwd_2/dose_edit_original.csv')\n"
        "s = np.load('/work/fwd_3/states_edited.npz')\n"
        f"print(f.shape, d.groupby('edit')['shift'].mean().round(2).to_dict(), s['x'].shape, s['has_insulin'].all(), '{REFUSE_MARKER}')"),
    E + "\n\n" + FW.format(json.dumps({"type": "factual", "model": "original", "split": "tuning", "stays": [999]})),
    E + "\n\n" + FW.format(json.dumps({"type": "factual", "model": "original", "split": "train", "stays": list(range(1010000000, 1010000101))})),
    E + "\n\n" + FW.format(json.dumps({"type": "factual", "model": "original", "split": "tuning", "stays": TU[:1]})) + "\n\n" + PY.format("1"),
    E + "\n\n" + PY.format("# hypothesis: a run past the time limit restarts the kernel\nimport time; time.sleep(3600)"),
    "I have nothing to run in this reply (dry run: this tests the reminder).",
    E + "\n\n" + PY.format("# after the restart: the edit is still on disk\nedited()"),
    "FINAL ANSWER\nDry run: I kept the 1.2x edit of encode_action.mlp2.2.weight. It was not chosen to fix anything; this is a plumbing test.",
]
BLACK = [
    E + "\n\n" + PY.format("# hypothesis: no weights are mounted, names and shapes are\nimport os\n"
                           "print(os.path.exists('/model/model.safetensors'), sorted(os.listdir('/model')), len(NAMES), SHAPES['core.c_out.weight'], ops())"),
    E + "\n\n" + FW.format(json.dumps({"type": "states", "model": "original", "split": "tuning", "stays": TU, "bundles": "insulin"})),
    E + "\n\n" + PY.format("# hypothesis: a direction from the states makes a rank-1 operation\nimport numpy as np\n"
                           "s = np.load('/work/fwd_1/states_original.npz'); v = s['x'][:, :192].mean(0)\n"
                           "print(rank1('encode_action.mlp2.2.weight', v, 0.5, 'out')); print(scale('core.c_out.weight', 1.01))\n"
                           "print(rank1('core.b_in.weight', v, -0.2, 'in'))\nops()"),
    E + "\n\n" + FW.format(json.dumps({"type": "dose_edit", "model": "both", "split": "train", "stays": TR})),
    E + "\n\n" + PY.format("# remove the last operation\nprint(undo()); ops()"),
    "FINAL ANSWER\nDry run: a rank-1 update of encode_action.mlp2.2.weight and a 1.01 scale of core.c_out.weight; a plumbing test.",
]
SCRIPTS = {"white": WHITE, "black": BLACK}


def reply(key, arm, prompt, state_dir):
    """the next scripted reply for this run (pilot prompts get the two pilot answers)"""
    if "BLUEBIRD" in prompt:
        return "Dry-run stand-in model by nobody\nBLUEBIRD"
    if "previous reply" in prompt and "word" in prompt:
        return "BLUEBIRD"
    p = Path(state_dir) / f"{key}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    st = json.loads(p.read_text()) if p.exists() else {"i": 0}
    i, arm = st["i"], arm or st.get("arm") or arm_of(prompt)   # the stand-in CLIs learn the arm from the first message
    if REFUSE_MARKER in prompt and not prompt.startswith("[The output"):
        return None                                          # the stand-in content filter refuses this message
    p.write_text(json.dumps({"i": i + 1, "arm": arm}))
    script = SCRIPTS[arm]
    return script[min(i, len(script) - 1)]


def arm_of(prompt):
    return "black" if "The weight values themselves are not available" in prompt else "white"


class LocalResponder:
    """the fake agent in-process (harness --responder fake)"""

    def __init__(self, run_dir):
        self.state = Path(run_dir) / "fake_state"
        self.arm = json.loads((Path(run_dir) / "state.json").read_text())["arm"]

    def ask(self, run, turn, prompt, thread):
        t0 = time.time()
        text = reply(run, self.arm, prompt, self.state)
        if text is None:
            return "", {"ok": False, "refused": True, "thread_id": f"local-{run}", "tool_free": True, "wall_s": 0}
        return text, {"ok": True, "refused": False, "thread_id": f"local-{run}", "tool_free": True,
                      "wall_s": round(time.time() - t0, 3), "models": ["fake"]}


def stub_state_dir():
    return os.environ.get("AWR_FAKE_STATE", "/tmp/awr_fake_state")
