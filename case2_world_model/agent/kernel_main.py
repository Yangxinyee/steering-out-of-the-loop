"""The agent's persistent Python kernel (adapted from case 1's agent kit). It runs inside the sandbox that agent/kernel.py
starts (no network, no GPU; /model and /data read-only, /work writable) and keeps one namespace across code runs, like a
notebook. Gradients are off.

Defined in advance (the task message lists them):
  both arms   NAMES, SHAPES, VOCAB (vocab.json), and the records at /data/train.csv and /data/tuning.csv
  white box   load(name), load_original(name), save_edit(name, tensor), drop_edit(name), edited(); the weights are in
              /model/model.safetensors; edits are kept as one float32 safetensors file per tensor in /work/edited/
  black box   no weight values; edits are operations kept in /work/ops.json and applied outside the sandbox:
              scale(name, c), rank1(name, vector, alpha, side="out"), ops(), undo(), clear_ops()

Protocol (agent/kernel.py): one JSON line {"code": ...} on stdin, one JSON line {"output": ...} back on the original
stdout. While the code runs, file descriptors 1 and 2 point at a capture file, so output written by C extensions is
caught too. If the code's last statement is an expression, its value is shown (as in a notebook, None is not).
"""
import ast
import json
import math
import os
import sys
import traceback

PROTO_OUT = os.fdopen(os.dup(1), "w", buffering=1)
PROTO_IN = sys.stdin
sys.stdin = open(os.devnull)
CAPTURE = "/tmp/kernel_capture.txt"
MODE = os.environ.get("AWR_MODE", "white")

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_grad_enabled(False)
SHAPES = {k: tuple(v) for k, v in json.load(open("/model/tensors.json")).items()}
NAMES = sorted(SHAPES)
VOCAB = json.load(open("/model/vocab.json"))
EDITED = "/work/edited"
OPS = "/work/ops.json"


def _check(name):
    if name not in SHAPES:
        raise KeyError(f"unknown tensor name: {name!r} (see NAMES)")


if MODE == "white":
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    _ORIG = safe_open("/model/model.safetensors", "pt")

    def _file(name):
        return os.path.join(EDITED, name + ".safetensors")

    def load_original(name):
        """the original tensor, as float32"""
        _check(name)
        return _ORIG.get_tensor(name).float()

    def load(name):
        """your current edit of the tensor if there is one, else the original, as float32"""
        _check(name)
        return load_file(_file(name))["w"].float() if os.path.exists(_file(name)) else load_original(name)

    def save_edit(name, tensor):
        """replace a tensor by your edited version (same shape, finite values)"""
        _check(name)
        t = torch.as_tensor(tensor).detach().to("cpu", torch.float32).contiguous()
        o = load_original(name)
        if tuple(t.shape) != tuple(o.shape):
            raise ValueError(f"save_edit refused: shape {tuple(t.shape)} differs from the original {tuple(o.shape)}")
        if not torch.isfinite(t).all():
            raise ValueError("save_edit refused: the tensor has NaN or infinite values")
        os.makedirs(EDITED, exist_ok=True)
        tmp = _file(name) + ".tmp"
        save_file({"w": t}, tmp)
        os.replace(tmp, _file(name))
        rel = float((t - o).norm() / o.norm().clamp_min(1e-30))
        return f"saved edit of {name} (relative change {rel:.4%})"

    def drop_edit(name):
        """go back to the original tensor"""
        _check(name)
        if os.path.exists(_file(name)):
            os.remove(_file(name))
            return f"dropped edit of {name}"
        return f"{name} had no edit"

    def edited():
        """the names of the tensors you have edited"""
        if not os.path.isdir(EDITED):
            return []
        return sorted(f[: -len(".safetensors")] for f in os.listdir(EDITED) if f.endswith(".safetensors"))

    API = {"load": load, "load_original": load_original, "save_edit": save_edit, "drop_edit": drop_edit, "edited": edited}

else:
    def _ops():
        return json.load(open(OPS)) if os.path.exists(OPS) else []

    def _write(ops_):
        tmp = OPS + ".tmp"
        with open(tmp, "w") as f:
            json.dump(ops_, f, indent=1)
        os.replace(tmp, OPS)

    def scale(name, c):
        """multiply a whole tensor by the number c (applied after the operations before it)"""
        _check(name)
        c = float(c)
        if not math.isfinite(c):
            raise ValueError("scale refused: c must be finite")
        o = _ops() + [{"op": "scale", "name": name, "c": c}]
        _write(o)
        return f"operation {len(o)}: scale {name} by {c:g}"

    def rank1(name, vector, alpha, side="out"):
        """a rank-1 update along a unit direction v (the vector, normalised): side "out" (v has shape[0] entries):
        W <- W + alpha * v (v^T W), which scales W's output component along v by 1 + alpha; side "in" (2-D only, v has
        shape[1] entries): W <- W + alpha * (W v) v^T, the same on its input. For a 1-D tensor, side "out":
        w <- w + alpha * v (v . w)."""
        _check(name)
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        shape = SHAPES[name]
        n = shape[0] if side == "out" else (shape[1] if side == "in" and len(shape) == 2 else None)
        if n is None:
            raise ValueError("rank1 refused: side must be 'out', or 'in' for a 2-D tensor")
        if v.size != n:
            raise ValueError(f"rank1 refused: {name} {shape} needs a vector of length {n} for side {side!r}, not {v.size}")
        if not np.isfinite(v).all() or np.linalg.norm(v) == 0 or not math.isfinite(float(alpha)):
            raise ValueError("rank1 refused: the vector must be finite and non-zero, alpha finite")
        os.makedirs("/work/vectors", exist_ok=True)
        o = _ops()
        path = f"vectors/v{len(o) + 1:03d}.npy"
        np.save("/work/" + path, v / np.linalg.norm(v))
        o.append({"op": "rank1", "name": name, "vector": path, "alpha": float(alpha), "side": side})
        _write(o)
        return f"operation {len(o)}: rank-1 update of {name} ({side}), alpha {float(alpha):g}"

    def ops():
        """the operations so far, in the order they are applied"""
        return _ops()

    def undo():
        """remove the last operation"""
        o = _ops()
        if not o:
            return "no operations"
        last = o.pop()
        _write(o)
        return f"removed operation {len(o) + 1}: {last}"

    def clear_ops():
        """remove all operations (back to the original model)"""
        _write([])
        return "all operations removed"

    API = {"scale": scale, "rank1": rank1, "ops": ops, "undo": undo, "clear_ops": clear_ops}

NS = {"__name__": "__main__", "NAMES": NAMES, "SHAPES": SHAPES, "VOCAB": VOCAB, **API}


def run(code):
    tree = ast.parse(code, "<run>")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<run>", "exec"), NS)
    if last is not None:
        v = eval(compile(last, "<run>", "eval"), NS)
        if v is not None:
            print(repr(v))


def user_traceback(e):
    """the traceback without the kernel's own frames"""
    tb = e.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != "<run>":
        tb = tb.tb_next
    lines = traceback.format_exception(type(e), e, tb) if tb is not None else traceback.format_exception_only(type(e), e)
    if tb is None and not isinstance(e, SyntaxError):
        lines = ["Traceback (most recent call last):\n"] + lines
    return "".join(lines)


def main():
    saved = os.dup(1), os.dup(2)
    PROTO_OUT.write(json.dumps({"ready": True, "tensors": len(NAMES), "mode": MODE}) + "\n")
    for line in PROTO_IN:
        req = json.loads(line)
        cap = open(CAPTURE, "w", buffering=1)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(cap.fileno(), 1)
        os.dup2(cap.fileno(), 2)
        sys.stdout = sys.stderr = cap
        try:
            run(req["code"])
            ok = True
        except BaseException as e:  # noqa: BLE001  (SystemExit and KeyboardInterrupt from the code are its errors too)
            cap.write(user_traceback(e))
            ok = False
        cap.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        cap.close()
        with open(CAPTURE, errors="replace") as f:
            out = f.read(100_000)           # the harness shows at most 10,000 characters
        PROTO_OUT.write(json.dumps({"output": out, "ok": ok, "chars": os.path.getsize(CAPTURE)}) + "\n")


if __name__ == "__main__":
    main()
