"""The logit-bias hook, on CPU with a stand-in model (no vLLM needed).

python -m unittest discover -s tests -v
"""

import unittest

import torch

from logit_psv.engine import LogitBiasExtension

VOCAB = 1000


class FakeModel:
    def __init__(self):
        self.calls = 0

    def compute_logits(self, hidden_states):
        self.calls += 1
        return (hidden_states @ torch.ones(4, VOCAB)).to(torch.bfloat16)


class FakeRunner:
    def __init__(self):
        self.model = FakeModel()

    def get_model(self):
        return self.model


class FakeWorker(LogitBiasExtension):
    def __init__(self):
        self.model_runner = FakeRunner()


def expected_bias(seed, sigma):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return float(sigma) * torch.randn(VOCAB, dtype=torch.float32, generator=gen)


class LogitBiasTest(unittest.TestCase):
    def setUp(self):
        self.worker = FakeWorker()
        self.worker.clear_logit_bias()
        self.model = self.worker.model_runner.model
        self.hidden = torch.randn(3, 4)

    def test_bias_is_added_in_fp32(self):
        base = self.model.compute_logits(self.hidden)
        self.worker.set_logit_bias(123, 0.1)
        out = self.model.compute_logits(self.hidden)
        self.assertEqual(out.dtype, torch.float32)
        # every row (decoding position / prompt) gets the same bias, exactly
        self.assertTrue(torch.equal(out, base.float() + expected_bias(123, 0.1)))
        # a small bias survives: it would be rounded away on the bf16 grid of the logits
        self.assertFalse(torch.equal(out, (base + expected_bias(123, 0.1).bfloat16()).float()))

    def test_bias_is_persistent_and_member_specific(self):
        self.worker.set_logit_bias(5, 2.0)
        first, second = (
            self.model.compute_logits(self.hidden),
            self.model.compute_logits(self.hidden),
        )
        self.assertTrue(torch.equal(first, second))
        self.worker.set_logit_bias(6, 2.0)
        self.assertFalse(torch.equal(first, self.model.compute_logits(self.hidden)))
        self.worker.set_logit_bias(5, 2.0)  # re-arming a member reproduces it exactly
        self.assertTrue(torch.equal(first, self.model.compute_logits(self.hidden)))

    def test_clear_restores_the_frozen_model(self):
        base = self.model.compute_logits(self.hidden)
        self.worker.set_logit_bias(5, 2.0)
        self.model.compute_logits(self.hidden)
        self.worker.clear_logit_bias()
        out = self.model.compute_logits(self.hidden)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(out, base))

    def test_hook_installed_once(self):
        self.worker.set_logit_bias(1, 1.0)
        hooked = self.model.compute_logits
        self.worker.set_logit_bias(2, 1.0)
        self.assertIs(self.model.compute_logits, hooked)
        self.model.compute_logits(self.hidden)
        self.assertEqual(self.model.calls, 1)  # the original runs once per call


if __name__ == "__main__":
    unittest.main()
