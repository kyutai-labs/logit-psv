"""Submit the paper cells as whole-node Slurm jobs. Dry run unless --submit.

    python -m reproduce.submit [--submit] [--hours 8] [--sbatch="--partition=..."] [PATTERN...]

Cells are packed into jobs of at most --hours estimated hours (first-fit decreasing;
the estimate is the sampling cost of the paper's runs on 8 GPUs plus 25% for the
ensemble phase and model loads). A cell that alone exceeds --hours gets its own job
with a longer walltime. Every cell resumes where it stopped, so a job that runs out of
time is simply submitted again with the same command.
"""

import argparse
import math
import os
import subprocess

from reproduce.cells import expected, select

GPUS_PER_NODE = 8


def estimate_hours(cell):
    gpu_hours = expected()[cell["id"]]["gpu_hours"]
    return 1.25 * gpu_hours / GPUS_PER_NODE + 0.25


def pack(cells, capacity):
    jobs = []  # [hours, [cells]]
    for cell in sorted(cells, key=estimate_hours, reverse=True):
        hours = estimate_hours(cell)
        for job in jobs:
            if job[0] + hours <= capacity:
                job[0] += hours
                job[1].append(cell)
                break
        else:
            jobs.append([hours, [cell]])
    return jobs


def walltime(hours):
    minutes = max(60, int(math.ceil(1.5 * hours * 60 / 30) * 30))  # 1.5x margin, 30 min steps
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("patterns", nargs="*")
    p.add_argument("--submit", action="store_true", help="submit (default: print the commands)")
    p.add_argument("--hours", type=float, default=8.0, help="estimated hours per job")
    p.add_argument("--sbatch", default="", help="extra sbatch options, e.g. '--partition=gpu'")
    args = p.parse_args()
    os.makedirs("logs", exist_ok=True)
    jobs = pack(select(args.patterns), args.hours)
    total = 0.0
    for i, (hours, cells) in enumerate(jobs):
        total += hours
        cmd = ["sbatch", "--parsable", f"--job-name=logit-psv-{i}", f"--time={walltime(hours)}"]
        cmd += args.sbatch.split() + ["reproduce/node.sbatch"] + [c["id"] for c in cells]
        if args.submit:
            job = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
            print(f"job {job}: ~{hours:.1f} h, {len(cells)} cells")
        else:
            print(f"# ~{hours:.1f} h\n{' '.join(cmd)}")
    print(f"{len(jobs)} jobs, ~{total:.0f} node-hours" + ("" if args.submit else " (dry run)"))


if __name__ == "__main__":
    main()
