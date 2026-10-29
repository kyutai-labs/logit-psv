# Tasks

All prompts are rendered with the model's chat template (`add_generation_prompt=True`).
Generation is greedy. The selection set is the first 200 items of the task's training
data, as defined below. Selection uses `reward` on the selection items; the reported
accuracy uses `check` on the majority vote of the members' `extract`ed answers.

| Task | Source | Selection / test items | Max new tokens |
|---|---|---|---|
| Countdown | `countdown.json` of [VsonicV/es-at-scale@1775c72](https://github.com/VsonicV/es-at-scale/blob/1775c7286b9123b8764c4ee3301ccd9dfe853f7e/countdown/data/countdown.json) (SHA-256 checked) | puzzles 0–199 / 200–2199 (2000) | 1024 |
| GSM8K | `openai/gsm8k`, `main` | train 0–199 / test (1319) | 1024 |
| MATH-500 | `HuggingFaceH4/MATH-500` | problems 0–199 / 200–499 (300) | 2048 |
| OlympiadBench | `Hothan/OlympiadBench`, `OE_TO_maths_en_COMP` (674), shuffled with `random.Random(0)` | 0–199 / 200–673 (474) | 2048 |
| MBPP | `google-research-datasets/mbpp`, `full`: train + validation + test in order (964) | rows 0–199 / 200–963 (764) | 2048 |
| ROCStories | `shawon/rocstories-combined` (98,161 stories) | stories 0–199 / 200–98,160 (97,961) | 64 |
| USPTO-50K | `pingzhili/uspto-50k` at revision `8296db0` | train 0–199 / validation (1001) | 1024 |

The paper scores ROCStories on the first 1000 test stories, except for Qwen2.5-3B,
which is scored on all 97,961 (`--test-samples`; see `reproduce/cells.py`).

## Countdown

Combine the given numbers with `+ - * /` to reach the target. A system message asks the
model to reason first, and the prompt asks for `<think>` reasoning and an `<answer>`
formula.

- **Vote**: the value of the last `<answer>` formula, provided the formula uses exactly
  the given numbers (each at most 1000). Votes pool on the value, so `1+2` and `2+1`
  agree. Formulas are evaluated like Python's `eval` (operators, precedence, `**`), but
  an integer power above 2^20 bits is invalid rather than a hang.
- **Reward**: 0.1 × format adherence + 1.0 × a correct answer (at most 1.1).
- **Check**: |value − target| < 1e-5.

## GSM8K

The question, followed by `Let's think step by step and output the final answer after "####".`

- **Vote**: the number after the last `#### ` in the final 300 characters, else the last
  number there, with commas and `$` removed.
- **Reward**: 1 if either the `####` answer or the last number equals the ground truth.
- **Check**: string equality with the ground truth.

## MATH-500 and OlympiadBench

The problem, followed by `Let's think step by step and output the final answer after ####`
(OlympiadBench appends a period). OlympiadBench ground truths have their surrounding `$`
removed.

- **Vote**: the last balanced `\boxed{...}` in the final 500 characters. Otherwise, in
  order: unbalanced `\boxed`, inline math at a line end, `answer is ...`,
  `therefore/thus/hence/so ...`, `answer: ...`, `= ...`. The keyword patterns are
  unanchored, so `so` also fires inside `solution`.
- **Reward and check**: LaTeX/whitespace normalization and string equality, then a
  numeric comparison (`\frac{p}{q}` is parsed as a prefix), then equality of the bare
  lists of numbers. The last two steps are lenient: they accept sign errors and `3` for
  `3\pi`.

## MBPP

A system message, then the task text, the first three tests, and a request for
`<think>` reasoning and code in `<answer>` tags.

- **Vote**: the raw code string in the last `<answer>` block, else the last
  `` ```python `` fence, else the last plain fence. Identical code strings pool.
- **Reward and check**: the code passes every test. The setup code, the candidate and
  its tests run together in a fresh Python interpreter, capped at 4 GB of address
  space, 10 s of CPU time and 15 s of wall-clock time. A pass also requires a one-time token printed after
  the last test, so `sys.exit()` or `os._exit(0)` cannot fake one. This limits
  resources but is not a security sandbox.

## ROCStories

Five shuffled sentences labelled A–E. The model must output their order as letters.

- **Vote**: the first complete permutation of `ABCDE`, written with commas, spaces, or
  as contiguous letters, else the first five letters A–E. Anything else abstains.
- **Reward and check**: a lenient score, 0.6 × (fraction of positions right) + 0.4 ×
  (fraction of the four gold adjacent pairs in the right relative order). Every valid
  permutation scores above 0. "Accuracy" is the mean score.

## USPTO-50K

Classify a reaction SMILES (atom-map numbers removed) into one of ten named reaction
classes. The system message lists the classes, and the answer goes in `<answer>X</answer>`.

- **Vote**: the class number (1–10) in the last `<answer>` tag, else the last bare
  number 1–10.
- **Check**: integer equality with the class. The majority class is 31.9% of the test set.
