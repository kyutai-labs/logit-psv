"""The run pipeline (sample -> ensemble -> vote, resume, sharding) with a stand-in engine.

python -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from logit_psv import engine as engine_module
from logit_psv import run
from logit_psv.psv import make_population, rank_members, vote_accuracy


class FakeTask:
    max_tokens = 8

    def reward(self, text, item):
        return float(text == "1")

    def extract(self, text, item):
        return "" if text == "2" else text  # "2" abstains

    def check(self, answer, item):
        return answer == str(item % 2)


class FakeEngine:
    """Member (seed, sigma) answers prompt i with str((seed + i) % 3); the base says "1"."""

    generated = 0
    started = 0

    def __init__(self, *args, **kwargs):
        FakeEngine.started += 1
        self.member = None

    def set_member(self, seed, sigma):
        self.member = (seed, sigma)

    def clear_member(self):
        self.member = None

    def generate(self, prompts, max_tokens):
        FakeEngine.generated += len(prompts)
        if self.member is None:
            return ["1"] * len(prompts)
        return [str((self.member[0] + int(p[1:])) % 3) for p in prompts]


def fake_data(args):
    data = SimpleNamespace(
        train_items=list(range(10)),
        test_items=list(range(25)),
        max_tokens=8,
    )
    data.train_prompts = [f"p{i}" for i in data.train_items]
    data.test_prompts = [f"p{i}" for i in data.test_items]
    return FakeTask(), data


ARGV = ["--model", "fake/model", "--task", "gsm8k"]
ARGV += ["--population", "12", "--sigmas", "0.5,1.0", "--top-k", "1,3,5", "--gpus", "1"]


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "run")
        patches = [
            mock.patch.object(run, "load_data", fake_data),
            mock.patch.object(engine_module, "Engine", FakeEngine),
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        FakeEngine.generated = 0

    def tearDown(self):
        self.tmp.cleanup()

    def main(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            run.main(ARGV + ["--out", self.out, *extra])

    def results(self):
        with open(os.path.join(self.out, "results.json")) as f:
            return json.load(f)

    def test_pipeline_matches_direct_computation(self):
        self.main()
        task, data = fake_data(None)
        members = make_population(12, [0.5, 1.0], 42)
        rows = [
            {"idx": i, "train_reward": sum((s + j) % 3 == 1 for j in range(10)) / 10}
            for i, (s, _) in enumerate(members)
        ]
        ranked = rank_members(rows)[:5]
        answers = [
            [task.extract(str((members[r["idx"]][0] + j) % 3), j) for j in range(25)]
            for r in ranked
        ]
        results = self.results()
        self.assertEqual([s["idx"] for s in results["selected"]], [r["idx"] for r in ranked])
        for k in (1, 3, 5):
            self.assertEqual(
                results["accuracy"][str(k)], vote_accuracy(task, answers, data.test_items, k)
            )
        self.assertEqual(results["base"], {"train_reward": 1.0, "test_accuracy": 12 / 25})

    def test_resume_does_no_work(self):
        self.main()
        first = self.results()
        FakeEngine.generated = FakeEngine.started = 0
        self.main()
        self.assertEqual((FakeEngine.generated, FakeEngine.started), (0, 0))
        self.assertEqual(self.results(), first)

    def test_larger_k_extends_the_ensemble(self):
        self.main()
        FakeEngine.generated = 0
        self.main("--top-k", "1,3,5,8")  # three more members on the 25 test prompts
        self.assertEqual(FakeEngine.generated, 3 * 25)
        self.assertEqual(len(self.results()["selected"]), 8)
        self.assertIn("8", self.results()["accuracy"])

    def test_partial_score_line_is_rescored(self):
        self.main("--phase", "sample")
        path = os.path.join(self.out, "scores", "shard0-of-1.jsonl")
        with open(path) as f:
            lines = f.readlines()
        with open(path, "w") as f:
            f.writelines(lines[:-1])
            f.write(lines[-1][:10])  # a job killed mid-write
        FakeEngine.generated = 0
        self.main("--phase", "sample")
        self.assertEqual(FakeEngine.generated, 10)  # one member, 10 train prompts
        with open(path) as f:
            self.assertEqual(f.readlines(), lines)

    def test_sharding_does_not_change_scores(self):
        self.main("--phase", "sample")
        single = run.read_scores(SimpleNamespace(out=self.out), make_population(12, [0.5, 1.0], 42))
        task, data = fake_data(None)
        args = run.parse_args(ARGV + ["--out", os.path.join(self.tmp.name, "sharded")])
        run.check_config(args, data, create=True)
        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(3):
                run.phase_sample(args, task, data, FakeEngine(), i, 3)
        sharded = run.read_scores(args, make_population(12, [0.5, 1.0], 42))
        self.assertEqual(sharded, single)
        self.assertEqual(len(os.listdir(os.path.join(args.out, "scores"))), 3)

    def test_other_config_is_refused(self):
        self.main("--phase", "sample")
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            run.main(ARGV[:-6] + ["--population", "13", "--sigmas", "0.5,1.0", "--out", self.out])

    def test_default_run_dir(self):
        argv = ["--model", "org/m", "--task", "gsm8k", "--sigmas", "1"]
        out = run.parse_args(argv).out
        self.assertEqual(out, os.path.join("runs", "gsm8k-m-N1000-sigma1.0-tr200-teall-seed42"))
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            run.parse_args(argv + ["--sigmas", "0"])


if __name__ == "__main__":
    unittest.main()
