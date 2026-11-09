"""SVD analysis of the authors' p50c edit, from the weights alone. CPU only, no forward pass: the base checkpoint and the authors' edit (rebuilt bit for bit, sha256-checked).

The edit is on the MLP's down_proj of layers 15 and 19 (1024 x 3584), not on an attention o_proj: its 64 columns J are
64 MLP neurons, so the notes' head reading (§4.0) is replaced by its MLP counterpart. Per layer, with u the unit
"repetition" direction recovered from the change (the left singular vector of the rank-one change on J; its sign is not
identifiable from the edit, since -c u uᵀ W is the same for -u, and is set to agree with our mean-difference loop
direction), W_J = W0[:, J] and b = W_Jᵀ u:

  0  facts       tensor, columns, coefficient c (fitted), rank one, sha256 of the rebuilt tensor
  1  alignment   cos²(u, left singular vectors) of W_J and of W, alpha² = |U_Jᵀu|², write strength |b| / s1, against
                 random unit directions, random 64-neuron subsets and the 64 strongest writers of u
  2  spectrum    s(W_J), s(W) before and after; the closed form W_J'ᵀW_J' = W_JᵀW_J + (c² - 2c) b bᵀ; spectral shift
                 |s' - s| / |s|; principal angles of the top-k singular subspaces; intruder directions (top singular
                 vectors of the edited matrix with max |cos| to every base singular vector below eps); c in 0,1,2,4,6,8
  3  input side  b in the right singular basis of W_J; the read directions of the 64 neurons weighted by b (up_proj
                 rows, gate_proj rows, through the post-attention norm's scale) as an approximate trigger; logit lens
                 of +-u and of the trigger (final norm scale, tied embedding) -- an approximation at a middle layer
  4  two edits   |cos(u15, u19)|, overlap of the two 64-neuron write subspaces, the cached mean-difference loop
                 directions (results/runs/meandiff, med_md), each against a random control

    nice -n 10 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m wsa.svd_edit --base ckpt/base --out results/svd_edit
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

from wsa.edits import LAYERS, NAME, TP, WFILE, authors_delta

sys.path.insert(0, TP)
import st_numpy as S  # noqa: E402

CS = (0, 1, 2, 4, 6, 8)
KS = (1, 4, 16, 64)
EPS = (0.3, 0.5, 0.7, 0.9)
N_RAND = 2000
P = "model.language_model."


def f64(bits):
    return S.bf16_to_f32(bits).astype(np.float64)


def unit(v):
    return v / np.linalg.norm(v)


def pct(x, sample):
    """share of the random sample at or below x (as a percentile)"""
    return float(100 * np.mean(np.asarray(sample) <= x))


def cum_cos2(U, u, ks):
    c2 = (U.T @ u) ** 2
    return c2, {k: float(c2[:k].sum()) for k in ks if k <= len(c2)}


def principal_cos(A, B):
    """cosines of the principal angles between the column spaces of orthonormal A and B"""
    return np.clip(np.linalg.svd(A.T @ B, compute_uv=False), 0, 1)


def intruders(U_new, U_base, k, eps):
    m = np.abs(U_new[:, :k].T @ U_base).max(1)
    return {str(e): int((m < e).sum()) for e in eps}, [round(float(x), 3) for x in m[:min(k, 8)]]


def top_tokens(vecs, tok, n=12):
    """vecs: (V,) scores -> the n highest and lowest tokens"""
    hi, lo = np.argsort(-vecs)[:n], np.argsort(vecs)[:n]
    show = lambda ids: [tok.decode([int(i)]) for i in ids]
    return {"top": show(hi), "bottom": show(lo)}


def main(a):
    rng = np.random.default_rng(0)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.base, "tokenizer.json"))
    names = [NAME.format(L) for L in LAYERS]
    names += [f"{P}layers.{L}.mlp.{p}_proj.weight" for L in LAYERS for p in ("gate", "up")]
    names += [f"{P}layers.{L}.post_attention_layernorm.weight" for L in LAYERS]
    names += [f"{P}norm.weight", f"{P}embed_tokens.weight"]
    T = S.load(os.path.join(a.base, WFILE), names)
    ad = authors_delta(a.base)                     # u (sign arbitrary), columns J, from the bit-exact rebuild
    man = json.load(open(os.path.join(TP, "p50c_edit", "manifest.json")))
    import apply_edit as AE
    z = np.load(os.path.join(TP, "p50c_edit", "edit.npz"))
    md = {k: np.load(os.path.join(a.runs, k, "dirs.npz")) for k in ("meandiff", "med_md")}
    E = S.bf16_to_f32(T[f"{P}embed_tokens.weight"][1])                 # tied: also the unembedding
    fin = 1.0 + S.bf16_to_f32(T[f"{P}norm.weight"][1]).astype(np.float64)
    out = {"layers": {}, "notes": {}}
    U_J_of, u_of = {}, {}

    for i, L in enumerate(LAYERS):
        name = NAME.format(L)
        t = man["tensors"][i]
        assert t["name"] == name
        W = f64(T[name][1])
        e_bits = AE.rebuild(T[name][1], t, z, i)
        sha = hashlib.sha256(np.ascontiguousarray(e_bits).tobytes()).hexdigest()
        We = f64(e_bits)
        d = ad[name]
        J, u = d["cols"], d["u"]
        if u @ md["meandiff"][f"u{L}"] < 0:          # sign set to agree with the loop direction (loop minus productive)
            u = -u
        u_of[L] = u
        R = {}

        # ---- 0 facts
        WJ, dJ = W[:, J], (We - W)[:, J]
        G = np.outer(u, u @ WJ)                                         # u uᵀ W_J
        c_fit = float(-(dJ * G).sum() / (G * G).sum())
        resid = np.linalg.norm(dJ + c_fit * G) / np.linalg.norm(dJ)
        runs = int(1 + np.sum(np.diff(J) > 1))
        R["facts"] = {
            "tensor": name, "shape": list(W.shape), "n_cols": int(len(J)), "cols_first_last": [int(J[0]), int(J[-1])],
            "contiguous_runs": runs, "sha256_matches_evaluated": sha == t["edited_sha256"],
            "c_fitted": round(c_fit, 4), "rank1_residual_rel": float(resid), "sv2_over_sv1": d["sv_ratio"],
            "rel_change_fro": float(np.linalg.norm(We - W) / np.linalg.norm(W))}
        wu = W.T @ u                                                    # every neuron's write along u
        top64 = np.argsort(-np.abs(wu))[:64]
        md_top = np.argsort(-md["meandiff"][f"score{L}"])[:64]
        rank_abs = np.argsort(np.argsort(-np.abs(wu)))                 # 0 = strongest |write|
        R["facts"].update({
            "J_in_top64_abs_write": int(len(np.intersect1d(J, top64))),
            "J_rank_abs_write_median": float(np.median(rank_abs[J])), "J_rank_abs_write_max": int(rank_abs[J].max()),
            "J_sign_of_write": {"positive": int((wu[J] > 0).sum()), "negative": int((wu[J] < 0).sum())},
            "J_overlap_meandiff_top64": int(len(np.intersect1d(J, md_top))), "random_overlap_expected": 64 * 64 / W.shape[1]})

        # ---- 1 alignment
        UJ, sJ, VJt = np.linalg.svd(WJ, full_matrices=False)
        UW, sW, VWt = np.linalg.svd(W, full_matrices=False)
        U_J_of[L] = UJ
        c2J, cumJ = cum_cos2(UJ, u, KS)
        c2W, cumW = cum_cos2(UW, u, KS + (256,))
        b = WJ.T @ u
        rand_u = np.array([unit(rng.standard_normal(W.shape[0])) for _ in range(N_RAND)])
        a2_rand_u = ((rand_u @ UJ) ** 2).sum(1)
        a2_sub, wr_sub = [], []
        for _ in range(N_RAND):
            Js = rng.choice(W.shape[1], 64, replace=False)
            Us = np.linalg.svd(W[:, Js], full_matrices=False)[0]
            a2_sub.append(float(((Us.T @ u) ** 2).sum()))
            wr_sub.append(float(np.linalg.norm(W[:, Js].T @ u)))
        Ut = np.linalg.svd(W[:, top64], full_matrices=False)[0]
        R["alignment"] = {
            "W_J": {"cos2_top8": [round(float(x), 4) for x in c2J[:8]], "cum": cumJ, "alpha2": cumJ[64],
                    "argmax_sv": int(np.argmax(c2J)), "write_strength_b_over_s1": float(np.linalg.norm(b) / sJ[0])},
            "W": {"cos2_top8": [round(float(x), 4) for x in c2W[:8]], "cum": cumW,
                  "weighted_mean_index": float((np.arange(len(c2W)) * c2W).sum()), "argmax_sv": int(np.argmax(c2W)),
                  "write_strength_Wtu_over_s1": float(np.linalg.norm(wu) / sW[0])},
            "baselines": {
                "random_u_alpha2_mean": float(a2_rand_u.mean()), "random_u_alpha2_p99": float(np.percentile(a2_rand_u, 99)),
                "expected_random_u_alpha2": 64 / W.shape[0],
                "random_64_neurons_alpha2_mean": float(np.mean(a2_sub)), "random_64_neurons_alpha2_p99": float(np.percentile(a2_sub, 99)),
                "J_alpha2_percentile_among_random_subsets": pct(cumJ[64], a2_sub),
                "random_64_neurons_write_mean": float(np.mean(wr_sub)), "J_write": float(np.linalg.norm(b)),
                "J_write_percentile_among_random_subsets": pct(float(np.linalg.norm(b)), wr_sub),
                "top64_writers_alpha2": float(((Ut.T @ u) ** 2).sum()),
                "top64_writers_write": float(np.linalg.norm(W[:, top64].T @ u))}}

        # ---- 2 spectrum
        sJe = np.linalg.svd(We[:, J], compute_uv=False)
        sWe = np.linalg.svd(We, compute_uv=False)
        cc = 6.0
        lam = np.sort(np.linalg.eigvalsh(WJ.T @ WJ + (cc * cc - 2 * cc) * np.outer(b, b)))[::-1]
        WJ6 = WJ - cc * np.outer(u, b)
        s6 = np.linalg.svd(WJ6, compute_uv=False)
        UJe, _, VJet = np.linalg.svd(We[:, J], full_matrices=False)
        UWe, _, VWet = np.linalg.svd(We, full_matrices=False)
        ang = {}
        for k in KS:
            ang[k] = {"W_left": [round(float(x), 4) for x in np.degrees(np.arccos(principal_cos(UW[:, :k], UWe[:, :k])))[[0, -1]]],
                      "W_right": [round(float(x), 4) for x in np.degrees(np.arccos(principal_cos(VWt[:k].T, VWet[:k].T)))[[0, -1]]]}
        intr_W, mx_W = intruders(UWe, UW, 16, EPS)
        intr_J, mx_J = intruders(UJe, UJ, 16, EPS)
        sweep = {}
        for c in CS:
            M = WJ - c * np.outer(u, b)
            Uc, sc, _ = np.linalg.svd(M, full_matrices=False)
            Mw = W.copy()
            Mw[:, J] = M
            sw = np.linalg.svd(Mw, compute_uv=False)
            sweep[str(c)] = {"W_J_s_top3": [round(float(x), 3) for x in sc[:3]], "W_J_fro2": float((M ** 2).sum()),
                             "cos2_u_top_left_sv": float((Uc[:, 0] @ u) ** 2), "W_s1": float(sw[0]),
                             "W_spectral_shift": float(np.linalg.norm(sw - sW) / np.linalg.norm(sW)),
                             "W_J_spectral_shift": float(np.linalg.norm(sc - sJ) / np.linalg.norm(sJ))}
        R["spectrum"] = {
            "W_J_s_top5_before": [round(float(x), 3) for x in sJ[:5]], "W_J_s_top5_after": [round(float(x), 3) for x in sJe[:5]],
            "W_s_top5_before": [round(float(x), 3) for x in sW[:5]], "W_s_top5_after": [round(float(x), 3) for x in sWe[:5]],
            "closed_form_max_rel_err_vs_formula": float(np.max(np.abs(np.sqrt(np.clip(lam, 0, None)) - s6) / s6[0])),
            "closed_form_max_rel_err_vs_bf16_edit": float(np.max(np.abs(np.sqrt(np.clip(lam, 0, None)) - sJe) / sJe[0])),
            "fro2_gain_W_J": float((sJe ** 2).sum() - (sJ ** 2).sum()), "fro2_gain_expected": float(24 * b @ b),
            "spectral_shift_W_J": float(np.linalg.norm(sJe - sJ) / np.linalg.norm(sJ)),
            "spectral_shift_W": float(np.linalg.norm(sWe - sW) / np.linalg.norm(sW)),
            "dW_spectral_norm": float(cc * np.linalg.norm(b)), "dW_spectral_over_W_s1": float(cc * np.linalg.norm(b) / sW[0]),
            "cos2_u_top_left_sv_after_W_J": float((UJe[:, 0] @ u) ** 2), "cos2_u_top_left_sv_after_W": float((UWe[:, 0] @ u) ** 2),
            "principal_angles_deg_first_last": ang,
            "intruders_top16_W": intr_W, "intruders_top16_W_maxcos_first8": mx_W,
            "intruders_top16_W_J": intr_J, "intruders_top16_W_J_maxcos_first8": mx_J, "c_sweep": sweep}

        # ---- 3 input side (approximation: each neuron's output a_j = silu(g_j·x)(p_j·x) taken as linear in its up read)
        bb = unit(b)
        c2V = (VJt @ bb) ** 2
        part = float(b @ b) ** 2 / float((b ** 4).sum())               # participation ratio of the 64 per-neuron writes
        lnw = 1.0 + S.bf16_to_f32(T[f"{P}layers.{L}.post_attention_layernorm.weight"][1]).astype(np.float64)
        Gr = f64(T[f"{P}layers.{L}.mlp.gate_proj.weight"][1])[J]
        Ur = f64(T[f"{P}layers.{L}.mlp.up_proj.weight"][1])[J]
        r_up = unit(lnw * (Ur.T @ b))
        r_gate = unit(lnw * (Gr.T @ np.abs(b)))
        lens = lambda v: (E @ (fin * v).astype(np.float32)).astype(np.float64)
        R["input_side"] = {
            "b_norm": float(np.linalg.norm(b)), "b_participation_ratio_of_64": part,
            "b_in_right_sv_basis_cos2_top8": [round(float(x), 4) for x in c2V[:8]],
            "b_right_sv_argmax": int(np.argmax(c2V)),
            "cos_r_up_u": float(r_up @ u), "cos_r_gate_u": float(r_gate @ u),
            "cos_r_up_meandiff": float(r_up @ unit(md["meandiff"][f"u{L}"])),
            "logit_lens_plus_u": top_tokens(lens(u), tok), "logit_lens_r_up": top_tokens(lens(r_up), tok),
            "embedding_receptor_r_up": top_tokens((E @ r_up.astype(np.float32)).astype(np.float64), tok)}
        R["notes"] = {"u_sign": "chosen so that u · meandiff u > 0; the edit itself is the same for -u"}
        out["layers"][str(L)] = R

    # ---- 4 the two edits
    u15, u19 = u_of[15], u_of[19]
    ru = np.array([unit(rng.standard_normal(1024)) for _ in range(N_RAND)])
    rc = np.abs((ru[: N_RAND // 2] * ru[N_RAND // 2:]).sum(1))
    ov = float((principal_cos(U_J_of[15], U_J_of[19]) ** 2).mean())
    W19 = f64(T[NAME.format(19)][1])
    ov_rand = [float((principal_cos(U_J_of[15], np.linalg.svd(W19[:, rng.choice(3584, 64, replace=False)],
                                                                    full_matrices=False)[0]) ** 2).mean()) for _ in range(500)]
    known = {}
    for key in ("meandiff", "med_md"):
        for L, uu in ((15, u15), (19, u19)):
            m = unit(md[key][f"u{L}"])
            known[f"{key}_u{L}"] = {"cos_same_layer": float(uu @ m),
                                    "cos_other_layer": float((u19 if L == 15 else u15) @ m),
                                    "random_abs_cos_p99": float(np.percentile(np.abs(ru @ m), 99))}
    out["two_edits"] = {
        "abs_cos_u15_u19": float(abs(u15 @ u19)), "random_abs_cos_mean": float(rc.mean()),
        "random_abs_cos_p99": float(np.percentile(rc, 99)),
        "subspace_overlap_J15_J19": ov, "subspace_overlap_random_64d_expected": 64 / 1024,
        "subspace_overlap_J15_vs_random_64_neurons_of_19_mean": float(np.mean(ov_rand)),
        "subspace_overlap_percentile": pct(ov, ov_rand),
        "known_directions": known,
        "interference": "not computed: the two edits sit in different layers whose neuron indices are unrelated; their "
                        "shared space is the residual stream, covered by the u cosine and the subspace overlap"}
    out["notes"]["subspace_overlap"] = "mean cos² of the 64 principal angles between the two write subspaces"
    out["notes"]["spectral_shift"] = "|s(edited) - s(base)|_2 / |s(base)|_2, as defined here"
    out["notes"]["intruders"] = "top-16 left singular vectors of the edited matrix whose largest |cos| with any base left singular vector is below eps"
    with open(a.out + ".json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False)[:20000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="ckpt/base")
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results/svd_edit")
    main(ap.parse_args())
