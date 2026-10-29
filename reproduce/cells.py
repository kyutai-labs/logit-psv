"""The paper's 27 logit-PSV cells: one per (model, task).

Every cell scores N=1000 members, each with the logit-bias scale sigma of its model,
selects on 200 items with seed 42, and reports the majority vote of the top K. The
cells give the logit-PSV column of the main-text and cross-scale tables and the
logit-PSV curves of the K-sweep figures:

- The test set is the task's full test set. ROCStories is the exception: the first
  1000 test stories, except for Qwen2.5-3B, which is scored on all 97,961.
- Qwen2.5-0.5B is run on six tasks. Its Countdown cell appears only in the K-sweep
  figure, and the paper has no USPTO result for it.
- The Countdown, GSM8K and MATH-500 cells of the K-sweep figure go up to K=128.

expected.json holds the paper's numbers for every cell (base and K-curve).
"""

import json
import os

MODELS = (
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "allenai/Olmo-3-7B-Instruct",
)
TASKS = ("countdown", "gsm8k", "math500", "olympiadbench", "mbpp", "rocstories", "uspto")

# The logit-bias scale is fixed per model, not tuned per task.
SIGMA = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.0,
    "Qwen/Qwen2.5-3B-Instruct": 1.0,
    "allenai/Olmo-3-7B-Instruct": 2.0,
}
NOT_RUN = {("Qwen/Qwen2.5-0.5B-Instruct", "uspto")}
TOP_K = [1, 2, 4, 5, 8, 10, 16, 20, 32, 50]
K_SWEEP = {  # cells whose K-curve goes up to K=128
    ("Qwen/Qwen2.5-0.5B-Instruct", "countdown"),
    *((m, t) for m in MODELS for t in ("gsm8k", "math500")),
    ("allenai/Olmo-3-7B-Instruct", "countdown"),
}


def _cell(model, task):
    test_samples = None
    if task == "rocstories" and model != "Qwen/Qwen2.5-3B-Instruct":
        test_samples = 1000
    return {
        "id": f"{model.split('/')[-1]}/{task}",
        "model": model,
        "task": task,
        "population": 1000,
        "sigma": SIGMA[model],
        "seed": 42,
        "train_samples": 200,
        "test_samples": test_samples,
        "top_k": TOP_K + ([64, 128] if (model, task) in K_SWEEP else []),
    }


CELLS = [_cell(m, t) for m in MODELS for t in TASKS if (m, t) not in NOT_RUN]
BY_ID = {c["id"]: c for c in CELLS}


def select(patterns):
    """Cells whose id contains every comma-separated part of one of `patterns`
    (e.g. "Olmo", "0.5B,gsm8k"); all cells if none given."""
    if not patterns:
        return list(CELLS)
    chosen = []
    for pattern in patterns:
        parts = pattern.split(",")
        chosen += [c for c in CELLS if all(p in c["id"] for p in parts) and c not in chosen]
    return chosen


def run_dir(cell, root="runs"):
    return os.path.join(root, cell["model"].split("/")[-1], cell["task"])


def cli_args(cell, root="runs"):
    """Arguments of `python -m logit_psv.run` for one cell."""
    args = [
        "--model", cell["model"],
        "--task", cell["task"],
        "--population", str(cell["population"]),
        "--sigmas", str(cell["sigma"]),
        "--seed", str(cell["seed"]),
        "--top-k", ",".join(map(str, cell["top_k"])),
        "--train-samples", str(cell["train_samples"]),
        "--out", run_dir(cell, root),
    ]  # fmt: skip
    if cell["test_samples"] is not None:
        args += ["--test-samples", str(cell["test_samples"])]
    return args


def expected():
    """{cell id: {"base": {...}, "accuracy": {K: acc}, "gpu_hours": h}}: the paper's
    numbers, and the GPU-hours the cell's sampling phase took on H100s."""
    with open(os.path.join(os.path.dirname(__file__), "expected.json")) as f:
        return json.load(f)
