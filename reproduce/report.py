"""Tabulate the reproduced cells and compare them with the paper's numbers.

    python -m reproduce.report [--root runs] [--k 50] [--tolerance 0] [PATTERN...]

Prints the base accuracy and the K-member logit-PSV accuracy of every finished cell
next to the paper's (reproduce/expected.json), and checks the base and the whole
K-curve of each finished cell. Exits with status 1 if any value differs by more than
--tolerance (absolute, as a fraction).
"""

import argparse
import json
import os

from reproduce.cells import expected, run_dir, select


def load(cell, root):
    path = os.path.join(run_dir(cell, root), "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("patterns", nargs="*")
    p.add_argument("--root", default="runs")
    p.add_argument("--k", default="50", help="ensemble size shown in the table")
    p.add_argument("--tolerance", type=float, default=0.0)
    args = p.parse_args()
    paper = expected()
    cells = select(args.patterns)
    finished = mismatched = 0
    print(f"| model | task | base | logit PSV K={args.k} | paper | diff | K-curve |")
    print("|---|---|---:|---:|---:|---:|---|")
    for cell in cells:
        ref = paper[cell["id"]]
        name, task = cell["model"].split("/")[-1], cell["task"]
        reported = 100 * ref["accuracy"][args.k]
        result = load(cell, args.root)
        if result is None:
            print(f"| {name} | {task} | | | {reported:.1f} | | not run |")
            continue
        finished += 1
        diffs = [
            abs(result["accuracy"][k] - v)
            for k, v in ref["accuracy"].items()
            if k in result["accuracy"]
        ]
        diffs.append(abs(result["base"]["test_accuracy"] - ref["base"]["test_accuracy"]))
        ok = max(diffs) <= args.tolerance
        mismatched += not ok
        ours = 100 * result["accuracy"][args.k]
        curve = "identical" if max(diffs) == 0 else f"max diff {100 * max(diffs):.2f} pt"
        print(
            f"| {name} | {task} | {100 * result['base']['test_accuracy']:.1f} | "
            f"{ours:.1f} | {reported:.1f} | {ours - reported:+.1f} | "
            f"{curve if ok else 'MISMATCH, ' + curve} |"
        )
    print(f"\n{finished}/{len(cells)} cells finished, {mismatched} outside the tolerance.")
    raise SystemExit(1 if mismatched else 0)


if __name__ == "__main__":
    main()
