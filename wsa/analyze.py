"""Part 1 table: every arm's accuracy, share cut at the cap and mean tokens on MATH-500 and GSM8K, its paired gain over
the base model (95% bootstrap interval over problems), rescues / regressions (same problem, same sample index), the
practice-problem screens that pick the decoding fix and the mean-difference coefficient, and the pre-set reading rules.

    python -m wsa.analyze --runs runs --out results/part1.json
"""
import argparse
import json
import os

import numpy as np

SETS = ("math500", "gsm8k")
BASE = "base/think35"
EDIT = "edit_authors/think35"
POOLS = {"random/think35": [f"random_s{i}/think35" for i in range(4)]}     # 4 random directions, 1 sample each
MEANDIFF_SCREEN = [f"meandiff_top64_a{a}/think35" for a in (1, 3, 6, 9, 12)]  # picked on practice only; tie -> smaller
ABLITERATION = "meandiff_topall_a1/think35"
SAME, CLEAR = 0.03, 0.05                                                     # the pre-set 3- and 5-point lines


DROP = set()        # test ids left out (--exclude: the templated near-neighbours of practice problems, wsa.contam)


def per_problem(path):
    return {r["id"]: r["correct"] for r in json.load(open(path)) if r["id"] not in DROP}


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(a, b):
    """gain of arm a over arm b: mean difference of per-problem accuracy, interval, rescues and regressions."""
    ids = [i for i in a if i in b and a[i] and b[i]]
    d = np.array([np.mean(a[i]) - np.mean(b[i]) for i in ids])
    resc = sum(x > y for i in ids for x, y in zip(a[i], b[i]))
    regr = sum(x < y for i in ids for x, y in zip(a[i], b[i]))
    return {"gain": float(d.mean()), "ci95": boot_ci(d), "rescues": int(resc), "regressions": int(regr), "n": len(ids)}


def load_arm(runs, arm, sets=SETS):
    """arm = '<run dir>/<prefix>' (e.g. 'base/think35'): its scores and per-problem results for each set present."""
    out = {}
    for s in sets:
        p = os.path.join(runs, f"{arm}_{s}.scores.json")
        if os.path.exists(p):
            out[s] = {"scores": json.load(open(p)),
                      "per": per_problem(os.path.join(runs, f"{arm}_{s}.per_problem.json"))}
    return out


def pool(parts):
    """several one-sample runs (e.g. 4 random directions) as one arm whose samples are the draws; keeps each draw's
    accuracy so the spread between directions is visible."""
    out = {}
    for s in SETS:
        got = [p[s] for p in parts if s in p]
        if len(got) != len(parts):
            continue
        tot = [g["scores"]["total"] for g in got]
        w = lambda key: float(np.average([g["scores"][key] for g in got], weights=tot))   # noqa: E731
        sc = {"acc": w("acc"), "cut_share": w("cut_share"), "mean_tokens": w("mean_tokens"),
              "samples": sum(g["scores"]["samples"] for g in got), "draws": [g["scores"]["acc"] for g in got]}
        per = {i: [c for g in got for c in g["per"].get(i, [])] for i in got[0]["per"]}
        out[s] = {"scores": sc, "per": per}
    return out


def row_of(d, base):
    row = {}
    for s, v in d.items():
        sc = v["scores"]
        acc = float(np.mean([c for cs in v["per"].values() for c in cs]))       # = sc["acc"] unless --exclude
        row[s] = {"acc": acc, "cut": sc["cut_share"], "mean_tokens": sc["mean_tokens"], "samples": sc["samples"]}
        for k in ("acc_cap4096", "acc_cap8192", "acc_cap16384", "draws"):
            if k in sc:
                row[s][k] = sc[k]
        if s in base and d is not base and sc["samples"] == base[s]["scores"]["samples"]:
            row[s]["vs_base"] = paired(v["per"], base[s]["per"])
        elif s in base and d is not base:           # different sample counts: compare problem means
            row[s]["vs_base"] = paired({i: [np.mean(c)] for i, c in v["per"].items() if c},
                                       {i: [np.mean(c)] for i, c in base[s]["per"].items() if c})
    return row


def practice(runs):
    """the screens on the practice problems (250 MATH + 250 GSM8K train problems, 1 sample): overall and per half."""
    out = {}
    for d in sorted(os.listdir(runs)):
        if not os.path.isdir(os.path.join(runs, d)):
            continue
        for f in sorted(os.listdir(os.path.join(runs, d))):
            if not f.endswith("_sel.scores.json"):
                continue
            arm = f"{d}/{f[:-len('_sel.scores.json')]}"
            per = per_problem(os.path.join(runs, d, f[:-len(".scores.json")] + ".per_problem.json"))
            halves = {h: float(np.mean([np.mean(c) for i, c in per.items() if i.startswith(h) and c]))
                      for h in ("math", "gsm8k")}
            out[arm] = {"acc": json.load(open(os.path.join(runs, d, f)))["acc"], **halves, "per": per}
    ref = out.get(BASE)
    for arm, v in out.items():
        if ref and arm != BASE:
            v["vs_base"] = paired(v["per"], ref["per"])
    return out


def best_of(prac, arms):
    have = [a for a in arms if a in prac]
    return max(have, key=lambda a: prac[a]["acc"]) if have else None


def rules(table, data, best_dec, best_md):
    """the pre-set reading rules. A = edit, B = best decoding fix (picked on practice), C = edit + B, all as gains
    over the base model on the same test set."""
    proto = best_dec.split("/", 1)[1]
    a_arm, b_arm, c_arm = EDIT, best_dec, f"edit_authors/{proto}"
    out = {"A": a_arm, "B": b_arm, "C": c_arm, "meandiff": best_md, "per_set": {}}
    for s in SETS:
        g = lambda arm: table.get(arm, {}).get(s, {}).get("vs_base", {}).get("gain")   # noqa: E731
        A, B, C = g(a_arm), g(b_arm), g(c_arm)
        if A is None or B is None:
            continue
        r = {"A": A, "B": B, "C": C, "A_minus_B": paired(data[a_arm][s]["per"], data[b_arm][s]["per"]),
             "B_share_of_A": B / A if A else None,
             "tokens_A": table[a_arm][s]["mean_tokens"], "tokens_B": table[b_arm][s]["mean_tokens"]}
        if C is not None:
            r["C_minus_B"] = paired(data[c_arm][s]["per"], data[b_arm][s]["per"])
        M = g(best_md) if best_md else None
        if M is not None:
            r["meandiff"] = M
            r["A_minus_meandiff"] = paired(data[a_arm][s]["per"], data[best_md][s]["per"])
        Ab = g(ABLITERATION)
        if Ab is not None:
            r["abliteration"] = Ab
        R = g("random/think35")
        if R is not None:
            r["random"] = R
        out["per_set"][s] = r
    ps = out["per_set"]
    if len(ps) == len(SETS):
        cheap = all(r["B"] >= r["A"] - SAME and r.get("C_minus_B", {}).get("gain", 1) < SAME for r in ps.values())
        value = [s for s, r in ps.items()
                 if (r["A_minus_B"]["gain"] >= CLEAR and r["A_minus_B"]["ci95"][0] > 0)
                 or r.get("C_minus_B", {}).get("gain", 0) >= CLEAR
                 or (abs(r["A"] - r["B"]) <= SAME and r["tokens_A"] <= 0.6 * r["tokens_B"])]
        out["verdict"] = ("expensive decoding fix" if cheap else
                          f"edit adds value ({', '.join(value)})" if value else
                          "in between: report the share of the edit's gain the decoding fix recovers")
        agent = {}
        for s, r in ps.items():
            if "A_minus_meandiff" in r:
                d = r["A_minus_meandiff"]["gain"]
                agent[s] = ("agent rediscovered the textbook recipe" if d <= SAME else
                            "agent's search adds" if d >= CLEAR else "in between")
            if "random" in r:
                agent[s + "_random"] = "non-specific" if r["random"] > r["A"] / 4 else "specific"
        out["agent"] = agent
    return out


def main(a):
    if a.exclude:
        DROP.update(x["id"] for v in json.load(open(a.exclude)).values() for x in v["flagged"])
    arms = sorted({os.path.join(d, f.rsplit("_", 1)[0]) for d in os.listdir(a.runs)
                   if os.path.isdir(os.path.join(a.runs, d))
                   for f in (x[:-len(".scores.json")] for x in os.listdir(os.path.join(a.runs, d))
                             if x.endswith(".scores.json")) if f.rsplit("_", 1)[-1] in SETS})
    data = {arm: load_arm(a.runs, arm) for arm in arms}
    for name, parts in POOLS.items():
        p = pool([data[x] for x in parts if x in data]) if all(x in data for x in parts) else {}
        if p:
            data[name] = p
    base = data.get(BASE, {})
    table = {arm: row_of(d, base) for arm, d in data.items()}
    prac = practice(a.runs)
    best_dec = best_of(prac, [x for x in prac if x.startswith("base/")])
    best_md = best_of(prac, MEANDIFF_SCREEN)
    verdict = rules(table, data, best_dec, best_md) if best_dec else {}
    json.dump({"test": table, "practice": {k: {x: y for x, y in v.items() if x != "per"} for k, v in prac.items()},
               "picked": {"decoding": best_dec, "meandiff": best_md}, "rules": verdict},
              open(a.out, "w"), indent=1)

    w = max(len(x) for x in list(table) + list(prac))
    print("practice problems (1 sample; overall, MATH half, GSM8K half, gain over base [95%])")
    for arm, v in prac.items():
        g = v.get("vs_base")
        gs = f"{100 * g['gain']:+5.1f} [{100 * g['ci95'][0]:+.1f},{100 * g['ci95'][1]:+.1f}]" if g else ""
        mark = " <- picked" if arm in (best_dec, best_md) else ""
        print(f"  {arm:<{w}} {100 * v['acc']:5.1f} {100 * v['math']:5.1f} {100 * v['gsm8k']:5.1f}  {gs}{mark}")
    print(f"\n{'test sets':<{w + 2}}" + "  ".join(f"{s:>7} acc  cut   tok    gain [95%]     " for s in SETS))
    for arm, row in table.items():
        cells = []
        for s in SETS:
            if s not in row:
                cells.append(" " * 44)
                continue
            r = row[s]
            g = r.get("vs_base")
            gs = f"{100 * g['gain']:+5.1f} [{100 * g['ci95'][0]:+.1f},{100 * g['ci95'][1]:+.1f}]" if g else ""
            cells.append(f"{100 * r['acc']:11.1f} {100 * r['cut']:4.0f}% {r['mean_tokens'] / 1000:5.1f}k {gs:<20}")
        print(f"  {arm:<{w}}" + "  ".join(cells))
    if verdict.get("per_set"):
        print(f"\nreading rules: A={verdict['A']}  B={verdict['B']}  C={verdict['C']}  meandiff={verdict['meandiff']}")
        for s, r in verdict["per_set"].items():
            ab = r["A_minus_B"]
            print(f"  {s}: A {100 * r['A']:+.1f}  B {100 * r['B']:+.1f}"
                  + (f"  C {100 * r['C']:+.1f}" if r["C"] is not None else "")
                  + f"  A-B {100 * ab['gain']:+.1f} [{100 * ab['ci95'][0]:+.1f},{100 * ab['ci95'][1]:+.1f}]"
                  + (f"  C-B {100 * r['C_minus_B']['gain']:+.1f}" if "C_minus_B" in r else "")
                  + (f"  meandiff {100 * r['meandiff']:+.1f}" if "meandiff" in r else "")
                  + (f"  random {100 * r['random']:+.1f}" if "random" in r else ""))
        print(f"  verdict: {verdict.get('verdict')}   agent: {verdict.get('agent')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", help="contam.json: recompute without the flagged test problems")
    main(ap.parse_args())
