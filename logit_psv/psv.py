"""Perturb-Select-Vote, the model-free parts: population, selection, voting, scoring.

A member is a pair (seed, sigma). Its logit bias, sigma * eps with eps ~ N(0, I_V)
drawn from a CUDA generator seeded with `seed` (see engine.py), is added to the
logits at every decoding step and for every input, and the member decodes greedily.
"""

import hashlib
import json
from collections import Counter

import numpy as np


def make_population(population, sigmas, seed):
    """The N members [(seed, sigma), ...], identical in every process: distinct
    seeds drawn from [0, 2^31), each with a sigma drawn uniformly from `sigmas`."""
    rng = np.random.default_rng(seed)
    seeds = rng.choice(2**31, size=population, replace=False)
    picks = rng.choice(sigmas, size=population)
    return [(int(s), float(g)) for s, g in zip(seeds, picks)]


def mean_reward(task, texts, items):
    """Selection score of one member: its mean reward over the selection items."""
    return float(np.mean([task.reward(t, d) for t, d in zip(texts, items)]))


def accuracy(task, answers, items):
    """Test score of one predictor (base model or a single member)."""
    return float(np.mean([bool(a) and task.check(a, d) for a, d in zip(answers, items)]))


def rank_members(rows):
    """Members sorted by train reward, best first. Ties go to the earlier population
    index, so the ranking never depends on how the population was sharded."""
    return sorted(rows, key=lambda r: (-r["train_reward"], r["idx"]))


def majority_vote(answers):
    """Plurality vote over one item's answers, given in rank order. Abstentions ("")
    do not vote; a tie goes to the answer that the best-ranked member gave first.
    Returns "" if every member abstained."""
    votes = [a for a in answers if a]
    # Counter keeps insertion order and most_common() sorts stably, so among equally
    # frequent answers the first one inserted (the best-ranked voter's) wins.
    return Counter(votes).most_common(1)[0][0] if votes else ""


def vote_accuracy(task, answers_by_rank, items, k):
    """Accuracy of the top-k members' majority vote. answers_by_rank[r][i] is the
    answer of the rank-r member on item i. check() may return a fraction (ROCStories)."""
    score = 0.0
    for i, item in enumerate(items):
        voted = majority_vote([answers_by_rank[r][i] for r in range(k)])
        if voted:
            score += float(task.check(voted, item))
    return score / len(items)


def digest(value):
    """SHA-256 of a JSON value. Used to fingerprint the train/test items and prompts
    of a run."""

    def default(v):
        if hasattr(v, "item"):
            try:
                return v.item()
            except (TypeError, ValueError):
                pass
        if isinstance(v, (set, frozenset)):
            return sorted(v, key=repr)
        return str(v)

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=default
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
