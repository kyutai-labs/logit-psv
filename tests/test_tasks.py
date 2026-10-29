"""Task semantics (CPU only, no download).

python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from logit_psv import tasks
from logit_psv.tasks import get_task


def fake_datasets(rows):
    """A stand-in `datasets` module whose load_dataset returns `rows`."""
    module = types.ModuleType("datasets")
    module.load_dataset = lambda *args, **kwargs: rows
    return mock.patch.dict(sys.modules, {"datasets": module})


class GSM8KTest(unittest.TestCase):
    def test_scoring(self):
        g = get_task("gsm8k")
        item = {"ground_truth": "150"}
        self.assertEqual(g.extract("blah #### 1,234"), "1234")
        self.assertEqual(g.extract("the answer is 42."), "42.")  # trailing dot kept
        self.assertEqual(g.extract("no numbers"), "")
        self.assertEqual(g.reward("#### 150", item), 1.0)
        self.assertEqual(g.reward("#### 7", item), 0.0)
        # a wrong strict answer still earns the reward when the last number is right...
        self.assertEqual(g.reward("#### 2.5 hours, which is 150 minutes", item), 1.0)
        # ...but the vote uses the strict answer
        self.assertEqual(g.extract("#### 2.5 hours, which is 150"), "2.5")
        self.assertEqual(g.reward("so it is 1,000", {"ground_truth": "1000"}), 1.0)
        self.assertTrue(g.check("150", item))
        self.assertFalse(g.check("150.0", item))


class MATH500Test(unittest.TestCase):
    def test_extract(self):
        m = get_task("math500")
        self.assertEqual(m.extract("so \\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")
        self.assertEqual(m.extract("\\boxed{a} junk \\boxed{42}"), "42")
        self.assertEqual(m.extract("The final answer is $3\\pi$."), "3\\pi")
        self.assertEqual(m.extract("answer \\boxed{\\frac{1}{2"), "\\frac{1")  # unbalanced
        # the keyword patterns are unanchored ("so" inside "solution")
        self.assertEqual(m.extract("We consider the solution is 5 apples"), "lution is 5 apples")

    def test_matcher(self):
        eq = get_task("math500")._equiv
        self.assertTrue(eq("\\frac{1}{2}", "0.5") and eq("\\dfrac{1}{2}", "\\frac{1}{2}"))
        self.assertTrue(eq("(-1, 6)", "(-1,6)") and eq("11,111", "11111"))
        self.assertTrue(eq("\\left( 3 \\right)", "(3)") and eq("Evelyn", "evelyn"))
        self.assertFalse(eq("1/3", "0.5") or eq("7", "8"))
        # the lenient comparisons accept sign errors and dropped constants
        self.assertTrue(eq("\\frac{24}{25}", "-\\frac{24}{25}") and eq("3", "3\\pi"))

    def test_check(self):
        m = get_task("math500")
        self.assertTrue(m.check("\\frac{1}{2}", {"ground_truth": "0.5"}))
        self.assertFalse(m.check("", {"ground_truth": ""}))


class CountdownTest(unittest.TestCase):
    item = {"numbers": [44, 19, 35], "ground_truth": 98.0}

    def test_scoring(self):
        c = get_task("countdown")
        good = "I will try.</think>\n<answer>(44 + 19) + 35</answer>"
        self.assertEqual(c.extract(good, self.item), "98")
        self.assertTrue(c.check("98", self.item))
        self.assertEqual(c.reward(good, self.item), 1.1)  # format + correct
        self.assertAlmostEqual(c.reward("<answer>44 + 19 + 35</answer>", self.item), 1.05)
        self.assertEqual(c.extract("<answer>44 + 19</answer>", self.item), "")  # wrong numbers
        self.assertEqual(c.extract("<answer>44 + 19 + 35 = 98</answer>", self.item), "98")
        self.assertEqual(c.extract("<answer>import os</answer>", self.item), "")  # charset
        # votes pool on the value, not the formula
        self.assertEqual(c.extract("<answer>35 + (44 + 19)</answer>", self.item), "98")
        # a formula on its own line is stripped
        own_line = "x</think>\n<answer>\n44 + 19 + 35\n</answer>"
        self.assertAlmostEqual(c.reward(own_line, self.item), 1.1)
        # ** passes the charset; a power tower is an invalid formula, not a hang
        four = {**self.item, "numbers": [44, 19, 35, 2]}
        self.assertEqual(c.extract("<answer>(44 - 19 - 35) ** 2</answer>", four), "100")
        tower = {"numbers": [9, 9, 9], "ground_truth": 9.0}
        self.assertEqual(c.extract("<answer>9**9**9</answer>", tower), "")
        self.assertEqual(c.reward("<answer>9**9**9</answer>", tower), 0.05)

    def test_data_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "countdown.json"), "w") as f:
                json.dump([{"numbers": [1, 2], "target": 3}], f)
            with mock.patch.object(tasks, "DATA_DIR", tmp):
                with self.assertRaisesRegex(ValueError, "sha256"):
                    get_task("countdown").load("train")


class OlympiadBenchTest(unittest.TestCase):
    def test_scoring(self):
        o = get_task("olympiadbench")
        item = {"ground_truth": "2"}
        self.assertEqual(o._clean_gt("$\\frac{1}{2}$"), "\\frac{1}{2}")
        self.assertEqual(o.reward("so \\boxed{2}", item), 1.0)
        self.assertEqual(o.reward("so \\boxed{3}", item), 0.0)
        self.assertEqual(o.reward("final answer is $2$", item), 1.0)

    def test_split_and_prompt(self):
        rows = [{"question": f"q{i}", "final_answer": [f"${i}$"]} for i in range(205)]
        with fake_datasets(rows):
            o = get_task("olympiadbench")
            train, test = o.load("train"), o.load("test")
        self.assertEqual((len(train), len(test)), (200, 5))
        questions = [x["messages"][0]["content"].split("\n")[0] for x in train + test]
        self.assertEqual(sorted(questions), sorted(f"q{i}" for i in range(205)))
        self.assertNotEqual(questions[:5], ["q0", "q1", "q2", "q3", "q4"])  # shuffled
        first = train[0]
        q = first["messages"][0]["content"].split("\n")[0]
        self.assertEqual(
            first["messages"][0]["content"],
            f"{q}\n\nLet's think step by step and output the final answer after ####.",
        )
        self.assertEqual(first["ground_truth"], q[1:])


class MBPPTest(unittest.TestCase):
    gt = {"test_list": ["assert add(2, 3) == 5", "assert add(0, 0) == 0"], "test_setup_code": ""}

    def test_scoring(self):
        m = get_task("mbpp")
        item = {"ground_truth": self.gt}
        good = "<think>sum</think>\n<answer>\ndef add(a, b):\n    return a + b\n</answer>"
        self.assertIn("def add", m.extract(good))
        self.assertEqual(m.reward(good, item), 1.0)
        self.assertEqual(
            m.reward("<answer>\ndef add(a, b):\n    return a - b\n</answer>", item), 0.0
        )
        self.assertEqual(m.reward("<answer>not code</answer>", item), 0.0)
        self.assertTrue(m.extract("```python\ndef add(a,b): return a+b\n```").startswith("def add"))
        self.assertEqual(m.extract("plain text, no code block"), "")

    def test_adversarial_code_fails_safely(self):
        m = get_task("mbpp")
        item = {"ground_truth": self.gt}
        self.assertEqual(m.reward("<answer>\nwhile True: pass\n</answer>", item), 0.0)
        exits = "<answer>\nimport os\ndef add(a, b):\n    return 0\nos._exit(0)\n</answer>"
        self.assertEqual(m.reward(exits, item), 0.0)
        rebound = "<answer>\nimport sys\nsys.exit = lambda *a: None\ndef add(a, b):\n    return 0\n</answer>"
        self.assertEqual(m.reward(rebound, item), 0.0)

    def test_slow_code_within_the_limit_passes(self):
        slow = "import time\ndef add(a, b):\n    time.sleep(1.5)\n    return a + b"
        self.assertTrue(tasks.MBPP._passes(slow, self.gt))


class ROCStoriesTest(unittest.TestCase):
    item = {"ground_truth": {"gold_labels": ["B", "C", "E", "D", "A"], "gold": "BCEDA"}}

    def test_scoring(self):
        r = get_task("rocstories")
        self.assertEqual(r.extract("The order is B,C,E,D,A."), "BCEDA")
        self.assertEqual(r.extract("answer: B C E D A"), "BCEDA")
        self.assertEqual(r.extract("BCEDA"), "BCEDA")
        self.assertEqual(r.extract("A,B,C"), "")
        self.assertEqual(r.extract("A,A,B,C,D"), "")
        self.assertEqual(r.check("BCEDA", self.item), 1.0)
        self.assertTrue(0.0 < r.check("ACEDB", self.item) < 1.0)  # lenient partial credit
        self.assertEqual(r.reward("B,C,E,D,A", self.item), 1.0)
        # the first complete order wins
        text = "First guess A, B, C, D, E. On reflection the order is B, C, E, D, A."
        self.assertEqual(r.extract(text), "ABCDE")

    def test_split(self):
        rows = [
            {"title": f"t{i}", "shuffled_sentences": list("abcde"), "gold_order": [0, 1, 2, 3, 4]}
            for i in range(205)
        ]
        with fake_datasets(rows):
            r = get_task("rocstories")
            train, test = r.load("train"), r.load("test")
        self.assertEqual((len(train), len(test)), (200, 5))
        self.assertIn("Title: t0\n", train[0]["messages"][-1]["content"])
        self.assertIn("Title: t200\n", test[0]["messages"][-1]["content"])


class USPTOTest(unittest.TestCase):
    def test_extract(self):
        u = get_task("uspto")
        item = {"ground_truth": "7"}
        self.assertEqual(u.extract("A reduction. <answer>7</answer>"), "7")
        self.assertEqual(u.extract("This is class 7"), "7")  # bare-number fallback
        self.assertEqual(u.extract("<answer>11</answer>"), "")  # labels are 1..10
        self.assertEqual(u.reward("<answer>7</answer>", item), 1.0)
        self.assertEqual(u.reward("<answer>3</answer>", item), 0.0)


class GetTaskTest(unittest.TestCase):
    def test_get_task(self):
        with self.assertRaises(KeyError):
            get_task("arc")
        self.assertEqual(get_task("math500", train_samples=50).split_at, 50)
        limits = {name: get_task(name).max_tokens for name in tasks.TASKS}
        self.assertEqual(
            limits,
            {
                "countdown": 1024,
                "gsm8k": 1024,
                "math500": 2048,
                "olympiadbench": 2048,
                "mbpp": 2048,
                "rocstories": 64,
                "uspto": 1024,
            },
        )


if __name__ == "__main__":
    unittest.main()
