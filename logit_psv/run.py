"""Run logit PSV for one (model, task): sample -> ensemble -> vote.

    python -m logit_psv.run --model Qwen/Qwen2.5-0.5B-Instruct --task gsm8k \\
        --population 1000 --sigmas 0.5

Phases (all three by default; each one resumes where it stopped):

  sample    the greedy base model, then the train reward of every member
  ensemble  the test answers of the top-K members, K = the largest --top-k
  vote      the majority-vote accuracy for every K in --top-k (CPU only)

With several visible GPUs, sample and ensemble start one engine per GPU, each on a
disjoint slice of the members. A member's output does not depend on the slicing, so
the results are those of a single GPU.

Run directory (default runs/<task>-<model>-N<N>-sigma<...>-...):

  config.json             configuration, data and prompt fingerprints, versions
  base.json               base model: train reward, test accuracy, test answers
  scores/*.jsonl          one row per member: idx, seed, sigma, train_reward
  answers/rankRRRR.json   test answers of the member ranked R (0 = best)
  results.json            accuracy for each K, and the selected members
  logs/                   per-GPU worker logs (runs on several GPUs)
"""

import argparse
import glob
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from logit_psv.psv import (
    accuracy,
    digest,
    make_population,
    mean_reward,
    rank_members,
    vote_accuracy,
)
from logit_psv.tasks import TASKS, get_task

DEFAULT_TOP_K = "1,2,4,5,8,10,16,20,32,50"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m logit_psv.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--population", type=int, default=1000, help="N, the number of members")
    p.add_argument(
        "--sigmas",
        required=True,
        help="comma-separated logit-bias scales; each member draws one uniformly",
    )
    p.add_argument("--top-k", default=DEFAULT_TOP_K, help="ensemble sizes K to report")
    p.add_argument("--seed", type=int, default=42, help="population seed")
    p.add_argument("--train-samples", type=int, default=200, help="selection items")
    p.add_argument("--test-samples", type=int, default=None, help="default: the full test set")
    p.add_argument("--max-tokens", type=int, default=None, help="default: the task's limit")
    p.add_argument("--gpus", type=int, default=0, help="GPUs to use (default: all visible)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    p.add_argument("--phase", action="append", choices=["sample", "ensemble", "vote"])
    p.add_argument("--out", default=None, help="run directory")
    p.add_argument("--worker", default=None, help=argparse.SUPPRESS)  # internal: "i/n"
    args = p.parse_args(argv)

    args.sigmas = [float(s) for s in args.sigmas.split(",")]
    args.top_k = sorted({int(k) for k in args.top_k.split(",")})
    args.phase = args.phase or ["sample", "ensemble", "vote"]
    if args.population <= 0 or not args.sigmas or any(s <= 0 for s in args.sigmas):
        p.error("--population and every --sigmas value must be positive")
    if not args.top_k or args.top_k[0] <= 0:
        p.error("--top-k values must be positive")
    if args.out is None:
        args.out = os.path.join(
            "runs",
            f"{args.task}-{args.model.rstrip('/').split('/')[-1]}"
            f"-N{args.population}-sigma{','.join(map(str, args.sigmas))}"
            f"-tr{args.train_samples}-te{args.test_samples or 'all'}-seed{args.seed}",
        )
    return args


# --------------------------------------------------------------------------
# Data, prompts and the run configuration
# --------------------------------------------------------------------------


def load_split(task, split, n, tries=6):
    """task.load, retried: concurrent dataset loads from a shared Hugging Face cache
    can raise a transient OSError from its file lock."""
    for attempt in range(tries):
        try:
            return task.load(split, n)
        except OSError as exc:
            if attempt == tries - 1:
                raise
            print(f"   retrying {split} load after: {exc}", flush=True)
            time.sleep(1.5 * (attempt + 1))


def build_prompts(items, model):
    """Render the chat messages with the model's chat template (plain text if none)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    if tok.chat_template is None:
        return ["\n".join(m["content"] for m in d["messages"]) + "\n" for d in items]
    return [
        tok.apply_chat_template(d["messages"], add_generation_prompt=True, tokenize=False)
        for d in items
    ]


def load_data(args):
    task = get_task(args.task, args.train_samples)
    train_items = load_split(task, "train", args.train_samples)
    test_items = load_split(task, "test", args.test_samples)
    return task, SimpleNamespace(
        train_items=train_items,
        test_items=test_items,
        train_prompts=build_prompts(train_items, args.model),
        test_prompts=build_prompts(test_items, args.model),
        max_tokens=args.max_tokens or task.max_tokens,
    )


def run_config(args, data):
    """What defines a run: resuming into a directory requires all of it to match."""
    return {
        "config": {
            "model": args.model,
            "task": args.task,
            "population": args.population,
            "sigmas": args.sigmas,
            "seed": args.seed,
            "train_samples": args.train_samples,
            "test_samples": args.test_samples,
            "max_tokens": data.max_tokens,
        },
        "data": {
            name: {"count": len(value), "sha256": digest(value)}
            for name, value in (
                ("train_items", data.train_items),
                ("test_items", data.test_items),
                ("train_prompts", data.train_prompts),
                ("test_prompts", data.test_prompts),
            )
        },
    }


def provenance(model):
    versions = {"python": sys.version.split()[0]}
    for package in ("numpy", "torch", "transformers", "vllm", "datasets"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = None
    try:
        from huggingface_hub import try_to_load_from_cache

        path = try_to_load_from_cache(model, "config.json")
        if isinstance(path, str) and "/snapshots/" in path:
            revision = path.split("/snapshots/")[1].split("/")[0]
    except ImportError:
        pass
    commit = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"versions": versions, "model_revision": revision, "git_commit": commit}


def write_json(obj, path, **kwargs):
    """Atomic write: a killed job never leaves a truncated file behind."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kwargs)
    os.replace(tmp, path)


def check_config(args, data, create):
    """Create config.json, or check that the existing one describes this same run."""
    path = os.path.join(args.out, "config.json")
    current = run_config(args, data)
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        for section in ("config", "data"):
            if saved.get(section) != current[section]:
                diff = sorted(
                    k
                    for k in set(saved.get(section, {})) | set(current[section])
                    if saved.get(section, {}).get(k) != current[section].get(k)
                )
                raise SystemExit(f"{args.out} holds a different run ({section}: {diff})")
    elif create:
        os.makedirs(args.out, exist_ok=True)
        write_json({**current, "provenance": provenance(args.model)}, path, indent=2)
    else:
        raise SystemExit(f"{path} is missing")


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------


def shard_range(n_items, i, n):
    """Contiguous slice i of n of range(n_items)."""
    return range(i * n_items // n, (i + 1) * n_items // n)


def read_scores(args, population):
    """{idx: row} from every score file of the run directory. A partial last line (a
    killed job) is ignored; a row that disagrees with the population is an error."""
    rows = {}
    for path in sorted(glob.glob(os.path.join(args.out, "scores", "*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (row["seed"], row["sigma"]) != population[row["idx"]]:
                    raise SystemExit(f"{path}: row {row['idx']} is from another population")
                rows.setdefault(row["idx"], row)
    return rows


def ranked_members(args, population):
    rows = read_scores(args, population)
    if len(rows) < args.population:
        raise SystemExit(f"{len(rows)}/{args.population} members scored: finish sampling first")
    return rank_members(rows.values())[: min(args.top_k[-1], args.population)]


def _trim_partial_line(path):
    """Drop the unterminated last line a killed job may have left, before appending."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        data = f.read()
        if data and not data.endswith(b"\n"):
            f.truncate(data.rfind(b"\n") + 1)


def phase_sample(args, task, data, engine, i, n):
    base_path = os.path.join(args.out, "base.json")
    if i == 0 and not os.path.exists(base_path):
        texts = engine.generate(data.train_prompts, data.max_tokens)
        train_reward = mean_reward(task, texts, data.train_items)
        texts = engine.generate(data.test_prompts, data.max_tokens)
        answers = [task.extract(t, d) for t, d in zip(texts, data.test_items)]
        test_accuracy = accuracy(task, answers, data.test_items)
        write_json(
            {"train_reward": train_reward, "test_accuracy": test_accuracy, "answers": answers},
            base_path,
        )
        print(f"base: train reward {train_reward:.4f}, test accuracy {test_accuracy:.4f}")

    population = make_population(args.population, args.sigmas, args.seed)
    done = read_scores(args, population)
    todo = [idx for idx in shard_range(args.population, i, n) if idx not in done]
    print(f"sample: worker {i}/{n}, {len(todo)} members to score", flush=True)
    os.makedirs(os.path.join(args.out, "scores"), exist_ok=True)
    path = os.path.join(args.out, "scores", f"shard{i}-of-{n}.jsonl")
    _trim_partial_line(path)
    with open(path, "a") as f:
        for idx in todo:
            seed, sigma = population[idx]
            start = time.time()
            engine.set_member(seed, sigma)
            texts = engine.generate(data.train_prompts, data.max_tokens)
            engine.clear_member()
            reward = mean_reward(task, texts, data.train_items)
            row = {"idx": idx, "seed": seed, "sigma": sigma, "train_reward": reward}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(
                f"   member {idx} seed={seed} sigma={sigma:g} reward={reward:.4f} "
                f"({time.time() - start:.1f}s)",
                flush=True,
            )


def _answers_path(args, rank):
    return os.path.join(args.out, "answers", f"rank{rank:04d}.json")


def _load_answers(args, rank, row, n_test):
    """The stored answers of the rank-`rank` member, or None if absent or stale."""
    try:
        with open(_answers_path(args, rank)) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    same = all(saved.get(k) == row[k] for k in ("idx", "seed", "sigma"))
    return saved if same and len(saved.get("answers", ())) == n_test else None


def phase_ensemble(args, task, data, engine, i, n):
    population = make_population(args.population, args.sigmas, args.seed)
    ranked = ranked_members(args, population)
    os.makedirs(os.path.join(args.out, "answers"), exist_ok=True)
    print(f"ensemble: worker {i}/{n}", flush=True)
    for rank in shard_range(len(ranked), i, n):
        row = ranked[rank]
        if _load_answers(args, rank, row, len(data.test_items)) is not None:
            continue
        engine.set_member(row["seed"], row["sigma"])
        texts = engine.generate(data.test_prompts, data.max_tokens)
        engine.clear_member()
        answers = [task.extract(t, d) for t, d in zip(texts, data.test_items)]
        solo = accuracy(task, answers, data.test_items)
        write_json(
            {"rank": rank, **row, "test_accuracy": solo, "answers": answers},
            _answers_path(args, rank),
        )
        print(
            f"   rank {rank} idx={row['idx']} train={row['train_reward']:.4f} test={solo:.4f}",
            flush=True,
        )


def has_work(args, phase, data):
    """Whether `phase` still has something to generate (so finished phases cost no
    engine start when a run is resumed or extended to a larger K)."""
    population = make_population(args.population, args.sigmas, args.seed)
    if phase == "sample":
        base = os.path.exists(os.path.join(args.out, "base.json"))
        return not base or len(read_scores(args, population)) < args.population
    ranked = ranked_members(args, population)
    return any(
        _load_answers(args, rank, row, len(data.test_items)) is None
        for rank, row in enumerate(ranked)
    )


def phase_vote(args, task, data):
    population = make_population(args.population, args.sigmas, args.seed)
    ranked = ranked_members(args, population)
    answers = []
    for rank, row in enumerate(ranked):
        saved = _load_answers(args, rank, row, len(data.test_items))
        if saved is None:
            raise SystemExit(f"{_answers_path(args, rank)} missing or stale: run --phase ensemble")
        answers.append(saved["answers"])
    with open(os.path.join(args.out, "base.json")) as f:
        base = json.load(f)
    curve = {
        k: vote_accuracy(task, answers, data.test_items, k) for k in args.top_k if k <= len(ranked)
    }
    print(f"{args.task} / {args.model}: base {base['test_accuracy']:.4f}")
    for k, acc in curve.items():
        print(f"   K={k:<4d} {acc:.4f}  ({acc - base['test_accuracy']:+.4f})")
    with open(os.path.join(args.out, "config.json")) as f:
        config = json.load(f)["config"]
    write_json(
        {
            "config": config,
            "base": {k: base[k] for k in ("train_reward", "test_accuracy")},
            "accuracy": {str(k): acc for k, acc in curve.items()},
            "selected": [
                {"rank": r, **{k: row[k] for k in ("idx", "seed", "sigma", "train_reward")}}
                for r, row in enumerate(ranked)
            ],
        },
        os.path.join(args.out, "results.json"),
        indent=2,
    )


# --------------------------------------------------------------------------
# Orchestration: one engine per GPU
# --------------------------------------------------------------------------


def visible_gpus():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if devices:
        return [d for d in devices.split(",") if d]
    import torch

    return [str(d) for d in range(torch.cuda.device_count())]


def worker_argv(args, phase, i, n):
    argv = [
        "--model", args.model,
        "--task", args.task,
        "--population", str(args.population),
        "--sigmas", ",".join(map(str, args.sigmas)),
        "--top-k", ",".join(map(str, args.top_k)),
        "--seed", str(args.seed),
        "--train-samples", str(args.train_samples),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--phase", phase,
        "--out", args.out,
        "--worker", f"{i}/{n}",
    ]  # fmt: skip
    if args.test_samples is not None:
        argv += ["--test-samples", str(args.test_samples)]
    if args.max_tokens is not None:
        argv += ["--max-tokens", str(args.max_tokens)]
    return argv


def run_workers(args, phase, gpus):
    """Run `phase` as len(gpus) single-GPU processes and wait for all of them."""
    n = len(gpus)
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    procs = []
    for i, gpu in enumerate(gpus):
        log = open(os.path.join(args.out, "logs", f"{phase}.worker{i}.log"), "a")
        cmd = [sys.executable, "-m", "logit_psv.run", *worker_argv(args, phase, i, n)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
        time.sleep(2)  # stagger the dataset-cache and model loads
    print(f"{phase}: {n} workers, logs in {args.out}/logs/", flush=True)
    failed = []
    for i, (proc, log) in enumerate(procs):
        if proc.wait() != 0:
            failed.append(i)
        log.close()
    if failed:
        raise SystemExit(f"{phase}: workers {failed} failed, see {args.out}/logs/")


def main(argv=None):
    args = parse_args(argv)
    task, data = load_data(args)
    gpu_phases = [p for p in args.phase if p != "vote"]

    if args.worker is not None:  # one GPU, one slice of one phase
        i, n = map(int, args.worker.split("/"))
        check_config(args, data, create=False)
        from logit_psv.engine import Engine

        engine = Engine(args.model, args.seed, args.gpu_memory_utilization)
        for phase in gpu_phases:
            {"sample": phase_sample, "ensemble": phase_ensemble}[phase](
                args, task, data, engine, i, n
            )
        return

    check_config(args, data, create=True)
    print(
        f"{args.task}: {len(data.train_items)} train / "
        f"{len(data.test_items)} test items, N={args.population}, sigmas={args.sigmas}, "
        f"out={args.out}",
        flush=True,
    )
    gpus = visible_gpus()
    if args.gpus:
        gpus = gpus[: args.gpus]
    engine = None
    for phase in args.phase:
        if phase == "vote":
            phase_vote(args, task, data)
            continue
        if not has_work(args, phase, data):
            print(f"{phase}: already complete", flush=True)
            continue
        n_jobs = args.population if phase == "sample" else min(args.top_k[-1], args.population)
        workers = gpus[:n_jobs]
        if len(workers) > 1:
            run_workers(args, phase, workers)
            continue
        if engine is None:  # a single GPU: run in this process, one engine for all phases
            from logit_psv.engine import Engine

            engine = Engine(args.model, args.seed, args.gpu_memory_utilization)
        {"sample": phase_sample, "ensemble": phase_ensemble}[phase](args, task, data, engine, 0, 1)


if __name__ == "__main__":
    main()
