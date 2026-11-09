"""A fixed low-rank edit of the plain model's state, applied after every core step (amnesic test; steering, control 1).

The state after a bundle is x = [out (192), h (4 x 192)], the same 960 numbers the probe reads. An edit is
(mu, R, L, c): x' = x + ((x - mu) @ R) * c @ L^T. With R = L = U orthonormal it is an orthogonal edit along U.
  * amnesic removal (LEACE, Belrose et al., arXiv 2306.03819): rank 1, R = W p, L = W^+ p, c = -1, with W the whitening
    of the state and p the whitened cross-covariance with the concept -> no linear function of x' covaries with it;
  * random control: the same eraser for a random direction p, c = -1;
  * steering (control 1): U = one unit direction, c = alpha -> the component along it is scaled by 1 + alpha.
The map is the same at every bundle and every stay, so it is equivalent to a low-rank linear layer on the core's output.
Only active lanes are edited; with a gate code, only after the bundles holding that drug (the post-dose state).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

D_OUT, N_SLOTS, D = 192, 4, 192
D_STATE = D_OUT + N_SLOTS * D


def load(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    R = z["R"] if "R" in z else z["U"]
    L = z["L"] if "L" in z else z["U"]
    return {"mu": z["mu"].astype(np.float32), "R": R.astype(np.float32), "L": L.astype(np.float32), "c": z["c"].astype(np.float32)}


def save_oblique(path: str | Path, mu: np.ndarray, R: np.ndarray, L: np.ndarray, c: np.ndarray, **meta) -> None:
    np.savez(path, mu=np.asarray(mu, np.float32), R=np.asarray(R, np.float32).reshape(D_STATE, -1),
             L=np.asarray(L, np.float32).reshape(D_STATE, -1), c=np.asarray(c, np.float32).reshape(-1), **meta)


def save(path: str | Path, mu: np.ndarray, U: np.ndarray, c: np.ndarray, **meta) -> None:
    U = np.asarray(U, dtype=np.float32).reshape(D_STATE, -1)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-4), "U must be orthonormal"
    np.savez(path, mu=np.asarray(mu, np.float32), U=U, c=np.asarray(c, np.float32).reshape(-1), **meta)


def insulin_code(store_root: str | Path) -> int:
    import polars as pl
    codes = pl.read_parquet(Path(store_root) / "vocab" / "codes.parquet")
    return int(codes.filter(pl.col("code") == "SOURCE/inputevents/223258")["code_id"][0])


def apply(model, edit: dict[str, np.ndarray], gate_code: int | None = None) -> None:
    """Wrap ``model.core.step`` so that the output and new slots are edited. With ``gate_code`` only after the bundles that
    hold a WORLD atom of that code (the insulin bundles, where the probe's states were taken); otherwise after every bundle."""
    dev = next(model.parameters()).device
    mu, R, L, c = (torch.tensor(edit[k], device=dev) for k in ("mu", "R", "L", "c"))
    orig_step, orig_forward = model.core.step, model.forward
    ctx: dict = {}

    def forward(batch, state):
        if gate_code is not None:
            B, K = batch["bundle_mask"].shape
            g = torch.zeros(B * K, dtype=torch.bool, device=batch["bundle_mask"].device)
            sel = (batch["atom_code"] == gate_code) & (batch["atom_kind"] == 0)
            g[batch["atom_bundle"][sel]] = True
            ctx["gate"], ctx["k"] = g.view(B, K), 0
        return orig_forward(batch, state)

    def step(state, bundle, dt, active, extra=(), n_loops=None):
        new, out = orig_step(state, bundle, dt, active, extra, n_loops)
        B = out.shape[0]
        on = active.reshape(B).bool()
        if gate_code is not None:
            on = on & ctx["gate"][:, ctx["k"]]
            ctx["k"] += 1
        x = torch.cat([out.float(), new["h"].float().reshape(B, -1)], dim=1)
        x2 = x + (((x - mu) @ R) * c) @ L.T
        x2 = torch.where(on.reshape(B, 1), x2, x)
        out2 = x2[:, :D_OUT].to(out.dtype)
        h2 = x2[:, D_OUT:].reshape(new["h"].shape).to(new["h"].dtype)
        return {**new, "h": h2}, out2

    model.core.step = step
    model.forward = forward
    model._state_edit = {"rank": int(R.shape[1]), "c": edit["c"].tolist(), "gated": gate_code is not None}
