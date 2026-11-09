"""The mean-difference coefficient picked on the practice problems only: the highest practice accuracy, the first in the
given order on a tie (so a larger coefficient has to beat a smaller one, not match it). Prints the coefficient.

    python -m wsa.pick_alpha --runs runs --alphas 6 9 12
"""
import argparse
import json
import sys


def main(a):
    acc = {x: json.load(open(f"{a.runs}/{a.pattern.format(x)}.scores.json"))["acc"] for x in a.alphas}
    print("practice: " + "  ".join(f"alpha {x} {100 * v:.1f}%" for x, v in acc.items()), file=sys.stderr)
    print(max(a.alphas, key=lambda x: acc[x]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--alphas", nargs="+", required=True)
    ap.add_argument("--pattern", default="meandiff_top64_a{}/think35_sel", help="practice run per coefficient")
    main(ap.parse_args())
