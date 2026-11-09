"""Edited checkpoints of Qwen3.5-0.8B (CPU only). Each is a copy of the base folder with the edited tensors replaced
in its single safetensors file.

  authors           the authors' p50c edit, rebuilt bit for bit by their apply_edit.py (sha256-checked)
  random  --seed s  a random unit direction r orthogonal to the authors' u, on the authors' 64 columns per layer,
                    W = W0 - beta * r (r^T W0) with beta set so the change has the authors' Frobenius norm
  learned --dirs F    a learned edit (wsa.learn_edit): per layer in the file, W = W0 - alpha * u (gate ⊙ u^T W0)
  authors_c --c c    the authors' own u and 64 columns per layer at another coefficient (the c sweep): W = W0 - c u (uᵀW0) on those columns; c 6 is their edit up to bf16 rounding,
                    c 1 removes the u component, c 2 reflects it
  swap --dir agent|meandiff --cols agent|meandiff --dirs F --alpha a
                    E2: one source's direction on the other's 64 columns per layer (15, 19),
                    W = W0 - alpha u (uᵀW0) there; agent = the authors' u and columns, meandiff = our mean-difference
                    direction and its top-64 columns by score (the textbook edit of part 1)
  meandiff --dirs F --alpha a --top 64|all [--layers 15 19]
                    our mean-difference direction u (wsa.direction) and its own top-64 writer columns (or all
                    columns), W = W0 - alpha * u (u^T W0); alpha 1 on all columns = classic abliteration; --layers
                    picks any layers whose u and score the file holds (wsa.layer_traits writes all 24)

    python -m wsa.edits <arm> --base ckpt/base --out ckpt/<name> [...]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.join(os.path.dirname(HERE), "third_party", "lm_understand_logs")
sys.path.insert(0, TP)
import st_numpy as S  # noqa: E402

WFILE = "model.safetensors-00001-of-00001.safetensors"
LAYERS = (15, 19)
NAME = "model.language_model.layers.{}.mlp.down_proj.weight"


def wfile(base):
    """the checkpoint's one weight file: Qwen3.5's name, or the only *.safetensors (DeepSeek-R1-Distill-Qwen-1.5B)"""
    if os.path.exists(os.path.join(base, WFILE)):
        return WFILE
    fs = [f for f in os.listdir(base) if f.endswith(".safetensors")]
    assert len(fs) == 1, f"{base}: expected one weight file, found {fs}"
    return fs[0]


def name_of(base):
    """down_proj weight name pattern: Qwen3.5 nests the text model under language_model, Qwen2 (R1-Distill) does not"""
    cfg = json.load(open(os.path.join(base, "config.json")))
    return NAME if "text_config" in cfg else "model.layers.{}.mlp.down_proj.weight"


def copy_base(base, out):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(base):
        p = os.path.join(base, f)
        if os.path.isfile(p) and f != wfile(base):
            shutil.copyfile(p, os.path.join(out, f))


def authors_delta(base):
    """the authors' u (unit, per layer) and column mask, recovered from their edit: on the changed columns the change
    is -6 u (u^T W0), a rank-one matrix whose left singular vector is u."""
    sys.path.insert(0, TP)
    import apply_edit as AE
    man = json.load(open(os.path.join(TP, "p50c_edit", "manifest.json")))
    z = np.load(os.path.join(TP, "p50c_edit", "edit.npz"))
    names = [t["name"] for t in man["tensors"]]
    base_t = S.load(os.path.join(base, WFILE), names)
    out = {}
    for i, t in enumerate(man["tensors"]):
        b = base_t[t["name"]][1]
        e = AE.rebuild(b, t, z, i)
        w0, we = S.bf16_to_f32(b).astype(np.float64), S.bf16_to_f32(e).astype(np.float64)
        d = we - w0
        cols = np.where(np.abs(d).sum(0) > 0)[0]
        uu, ss, _ = np.linalg.svd(d[:, cols], full_matrices=False)
        out[t["name"]] = {"u": uu[:, 0], "cols": cols, "fro": float(np.linalg.norm(d)), "sv_ratio": float(ss[1] / ss[0])}
    return out


def write(base, out, new, info):
    copy_base(base, out)
    tensors = S.load(os.path.join(base, wfile(base)))
    for name, w in new.items():
        dt, b = tensors[name]
        assert dt == "BF16" and b.shape == w.shape
        tensors[name] = ("BF16", S.bf16_bits(w.astype(np.float32)).reshape(b.shape))
        info.setdefault("sha256", {})[name] = hashlib.sha256(np.ascontiguousarray(tensors[name][1]).tobytes()).hexdigest()
    S.save(os.path.join(out, wfile(base)), tensors, {"format": "pt"})
    json.dump(info, open(os.path.join(out, "wsa_edit.json"), "w"), indent=1)
    print(json.dumps(info))


def main(a):
    if a.arm == "authors":
        import apply_edit as AE
        copy_base(a.base, a.out)
        AE.main(os.path.join(a.base, WFILE), os.path.join(TP, "p50c_edit"), os.path.join(a.out, WFILE))
        json.dump({"arm": "authors"}, open(os.path.join(a.out, "wsa_edit.json"), "w"))
        return
    if a.arm == "learned":
        zl = np.load(a.dirs)
        layers = tuple(sorted(int(k[5:]) for k in zl.files if k.startswith("alpha")))
    else:
        layers = tuple(a.layers) if a.arm == "meandiff" else LAYERS
    names = [name_of(a.base).format(L) for L in layers]
    base_t = S.load(os.path.join(a.base, wfile(a.base)), names)
    w0s = {n: S.bf16_to_f32(base_t[n][1]).astype(np.float64) for n in names}
    new, info = {}, {"arm": a.arm}
    if a.arm == "authors_c":
        ad = authors_delta(a.base)
        info["c"] = a.c
        for n in names:
            u, cols = ad[n]["u"], ad[n]["cols"]
            w = w0s[n].copy()
            w[:, cols] -= a.c * np.outer(u, u @ w[:, cols])
            new[n] = w
            info[n] = {"n_cols": len(cols), "fro_change": float(np.linalg.norm(w - w0s[n])), "authors_fro_change": ad[n]["fro"]}
    elif a.arm == "swap":
        ad = authors_delta(a.base)
        z = np.load(a.dirs)
        info.update({"dir": a.dir, "cols": a.cols, "alpha": a.alpha, "dirs": a.dirs})
        for L, n in zip(LAYERS, names):
            u = ad[n]["u"] if a.dir == "agent" else z[f"u{L}"].astype(np.float64) / np.linalg.norm(z[f"u{L}"])
            cols = ad[n]["cols"] if a.cols == "agent" else np.argsort(-z[f"score{L}"])[:64]
            w = w0s[n].copy()
            w[:, cols] -= a.alpha * np.outer(u, u @ w[:, cols])
            new[n] = w
            info[n] = {"n_cols": len(cols), "fro_change": float(np.linalg.norm(w - w0s[n])), "authors_fro_change": ad[n]["fro"]}
    elif a.arm == "random":
        ad = authors_delta(a.base)
        rng = np.random.default_rng(a.seed)
        info["seed"] = a.seed
        for n in names:
            u, cols = ad[n]["u"], ad[n]["cols"]
            r = rng.standard_normal(u.shape)
            r -= (r @ u) * u
            r /= np.linalg.norm(r)
            w = w0s[n].copy()
            d = -np.outer(r, r @ w[:, cols])
            beta = ad[n]["fro"] / np.linalg.norm(d)
            w[:, cols] += beta * d
            new[n] = w
            info[n] = {"beta": beta, "n_cols": len(cols), "authors_sv_ratio": ad[n]["sv_ratio"]}
    elif a.arm == "meandiff":
        z = np.load(a.dirs)
        tc = json.load(open(os.path.join(a.base, "config.json")))
        tc = tc.get("text_config", tc)
        ad = authors_delta(a.base) if tc["hidden_size"] == 1024 else {}     # the authors edited the 0.8B only
        info.update({"alpha": a.alpha, "top": a.top, "dirs": a.dirs, "layers": list(layers)})
        for L, n in zip(layers, names):
            u = z[f"u{L}"].astype(np.float64)
            u /= np.linalg.norm(u)
            w = w0s[n].copy()
            cols = np.arange(w.shape[1]) if a.top == "all" else np.argsort(-z[f"score{L}"])[:int(a.top)]
            w[:, cols] -= a.alpha * np.outer(u, u @ w[:, cols])
            new[n] = w
            info[n] = {"fro_change": float(np.linalg.norm(w - w0s[n]))}
            if n in ad:                                    # the authors edited layers 15 and 19 only
                info[n].update(cos_with_authors_u=float(abs(u @ ad[n]["u"])), authors_fro_change=ad[n]["fro"],
                               col_overlap_with_authors=int(len(set(cols.tolist()) & set(ad[n]["cols"].tolist()))))
    elif a.arm == "learned":
        info["dirs"] = a.dirs
        for L, n in zip(layers, names):
            u = zl[f"u{L}"].astype(np.float64)
            u /= np.linalg.norm(u)
            g, al = zl[f"gate{L}"].astype(np.float64), float(zl[f"alpha{L}"])
            w = w0s[n].copy()
            w -= al * np.outer(u, g * (u @ w))
            new[n] = w
            info[n] = {"alpha": al, "n_open": int((g > 0.5).sum()), "gate_sum": float(g.sum()),
                       "fro_change": float(np.linalg.norm(w - w0s[n]))}
    write(a.base, a.out, new, info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=["authors", "authors_c", "swap", "random", "meandiff", "learned"])
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dirs")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--c", type=float, default=6.0, help="authors_c: the coefficient")
    ap.add_argument("--dir", choices=["agent", "meandiff"], help="swap: whose direction")
    ap.add_argument("--cols", choices=["agent", "meandiff"], help="swap: whose 64 columns per layer")
    ap.add_argument("--top", default="64")
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS), help="meandiff: the layers to edit")
    main(ap.parse_args())
