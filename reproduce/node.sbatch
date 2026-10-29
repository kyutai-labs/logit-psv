#!/bin/bash
# One whole node (8 GPUs) running paper cells one after another (reproduce/run.py).
# Submitted by reproduce/submit.py; adapt the resources to your cluster.
#SBATCH --job-name=logit-psv
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
exec .venv/bin/python -m reproduce.run "$@"
