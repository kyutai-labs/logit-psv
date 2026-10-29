"""Population, selection and voting (CPU only).

python -m unittest discover -s tests -v
"""

import unittest

from logit_psv.psv import (
    accuracy,
    digest,
    majority_vote,
    make_population,
    mean_reward,
    rank_members,
    vote_accuracy,
)


class FractionTask:
    """check() returns a score in [0, 1], as ROCStories does."""

    def check(self, answer, item):
        return {"right": 1.0, "close": 0.5}.get(answer, 0.0)

    def reward(self, text, item):
        return float(text == item)


class PopulationTest(unittest.TestCase):
    def test_paper_members(self):
        # (seed, sigma) of members 0, 1, 2 and N-1 of the paper's OLMo-3-7B population.
        members = make_population(1000, [2.0], 42)
        self.assertEqual(
            [members[i] for i in (0, 1, 2, 999)],
            [(1607623001, 2.0), (1728506961, 2.0), (1045090458, 2.0), (839508132, 2.0)],
        )

    def test_distinct_seeds_and_grid(self):
        members = make_population(5000, [0.25, 8.0], 7)
        self.assertEqual(len({s for s, _ in members}), 5000)
        self.assertTrue(all(0 <= s < 2**31 for s, _ in members))
        self.assertEqual({g for _, g in members}, {0.25, 8.0})
        self.assertTrue(all(type(s) is int and type(g) is float for s, g in members))


class SelectionTest(unittest.TestCase):
    def test_rank_ties_go_to_lower_index(self):
        rows = [
            {"idx": 3, "train_reward": 0.5},
            {"idx": 1, "train_reward": 0.5},
            {"idx": 2, "train_reward": 0.9},
            {"idx": 0, "train_reward": 0.1},
        ]
        self.assertEqual([r["idx"] for r in rank_members(rows)], [2, 1, 3, 0])

    def test_mean_reward_and_accuracy(self):
        task = FractionTask()
        self.assertEqual(mean_reward(task, ["a", "b", "c"], ["a", "x", "c"]), 2 / 3)
        # "" abstains and scores 0; fractional checks count as fractions
        self.assertEqual(accuracy(task, ["right", "close", ""], [None] * 3), 0.5)


class VoteTest(unittest.TestCase):
    def test_plurality_abstention_and_ties(self):
        self.assertEqual(majority_vote(["a", "b", "b"]), "b")
        self.assertEqual(majority_vote(["", "", "a"]), "a")  # abstentions do not vote
        self.assertEqual(majority_vote(["", ""]), "")
        # a tie goes to the answer the best-ranked member gave first
        self.assertEqual(majority_vote(["b", "a", "a", "b"]), "b")
        self.assertEqual(majority_vote(["", "a", "b"]), "a")

    def test_vote_accuracy(self):
        task = FractionTask()
        # ranks x items
        answers = [["right", "close", ""], ["wrong", "close", "right"], ["wrong", "x", ""]]
        items = [None, None, None]
        self.assertEqual(vote_accuracy(task, answers, items, 1), (1.0 + 0.5 + 0.0) / 3)
        # K=2: item 0 ties right/wrong -> rank 0's "right"; item 2 only rank 1 votes
        self.assertEqual(vote_accuracy(task, answers, items, 2), (1.0 + 0.5 + 1.0) / 3)
        self.assertEqual(vote_accuracy(task, answers, items, 3), (0.0 + 0.5 + 1.0) / 3)


class DigestTest(unittest.TestCase):
    def test_digest_is_canonical(self):
        # key order does not matter; the value is pinned because run configs record it
        self.assertEqual(
            digest({"b": [1, 2.5, "é"], "a": None}), digest({"a": None, "b": [1, 2.5, "é"]})
        )
        self.assertEqual(
            digest({"b": [1, 2.5, "é"], "a": None}),
            "21fddaeebc6c84f1b55624c41336b0e521adecf3e414f2eaa12789afd6167428",
        )


if __name__ == "__main__":
    unittest.main()
