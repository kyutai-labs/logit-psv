"""Run paper cells one after another, each as `python -m logit_psv.run`.

    python -m reproduce.run [--root runs] PATTERN...

A PATTERN selects the cells whose id contains every comma-separated part of it
(ids look like "Qwen2.5-0.5B-Instruct/gsm8k"): "Olmo", "gsm8k", "0.5B,gsm8k". Each
cell writes to <root>/<model>/<task> and resumes there if it was interrupted. A
failed cell does not stop the others.
"""

import argparse
import subprocess
import sys

from reproduce.cells import cli_args, select


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("patterns", nargs="+")
    p.add_argument("--root", default="runs", help="root directory of the runs")
    args = p.parse_args()
    failed = []
    for cell in select(args.patterns):
        print(f"===== {cell['id']} =====", flush=True)
        cmd = [sys.executable, "-m", "logit_psv.run", *cli_args(cell, args.root)]
        if subprocess.call(cmd) != 0:
            failed.append(cell["id"])
    if failed:
        raise SystemExit(f"failed cells: {failed}")


if __name__ == "__main__":
    main()
