"""Greedy generation with vLLM, plus the logit bias of one PSV member.

The model is never modified. `LogitBiasExtension` wraps the model's compute_logits so
that, while a member is armed, every logits row gets the member's bias added:

    logits' = logits.float() + sigma * eps,   eps = randn(V, generator=cuda_gen(seed))

The bias is the same at every decoding step and for every prompt. It is added in fp32:
vLLM returns bf16 logits and its sampler upcasts them anyway, whereas a bf16 sum would
round the bias to the logits' bf16 grid. With temperature 0 the sampler takes the
arg-max, so a member is a deterministic function of (seed, sigma).

The engine settings (bf16, eager mode, no prefix caching, one generate call over all
prompts) are those of the paper's runs; keep them to reproduce their outputs exactly.
"""

import os

import torch


class LogitBiasExtension:
    """vLLM worker extension, loaded by name (worker_extension_cls) and called through
    LLM.collective_rpc. Keep it importable without vLLM."""

    def _install_hook(self):
        runner = self.model_runner
        model = runner.get_model() if hasattr(runner, "get_model") else runner.model
        if getattr(model, "_psv_hooked", False):
            return
        original = model.compute_logits
        ext = self

        def compute_logits(hidden_states, *args, **kwargs):
            logits = original(hidden_states, *args, **kwargs)
            if logits is None or ext._psv_member is None:
                return logits
            return logits.float() + ext._psv_bias(logits.shape[-1], logits.device)

        model.compute_logits = compute_logits
        model._psv_hooked = True

    def _psv_bias(self, vocab, device):
        """sigma * eps for the armed member, drawn once and reused at every step."""
        key = (self._psv_member, vocab, str(device))
        if getattr(self, "_psv_cache_key", None) != key:
            seed, sigma = self._psv_member
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
            eps = torch.randn(vocab, dtype=torch.float32, device=device, generator=gen)
            self._psv_cache = float(sigma) * eps
            self._psv_cache_key = key
        return self._psv_cache

    def set_logit_bias(self, seed, sigma):
        self._install_hook()
        self._psv_member = (int(seed), float(sigma))
        return True

    def clear_logit_bias(self):
        self._psv_member = None
        return True


class Engine:
    """One vLLM engine on the visible GPU(s). generate() decodes greedily; while a member
    is armed with set_member(seed, sigma), its logit bias applies to every token."""

    def __init__(self, model, seed=42, gpu_memory_utilization=0.80):
        # In-process engine, so collective_rpc reaches the worker extension directly.
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
        from vllm import LLM

        self.seed = seed
        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            enforce_eager=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            worker_extension_cls="logit_psv.engine.LogitBiasExtension",
        )
        self.llm.collective_rpc("clear_logit_bias")

    def generate(self, prompts, max_tokens):
        from vllm import SamplingParams

        params = SamplingParams(temperature=0.0, seed=self.seed, max_tokens=max_tokens)
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def set_member(self, seed, sigma):
        self.llm.collective_rpc("set_logit_bias", args=(seed, sigma))

    def clear_member(self):
        self.llm.collective_rpc("clear_logit_bias")
