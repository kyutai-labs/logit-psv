"""The seven benchmarks: data loading, prompts, answer extraction, reward and check.

Every task provides

    load(split, n)      -> list of items {"messages": [...], "ground_truth": ..., ...}
    extract(text, item) -> the answer a member votes for ("" = invalid, abstains)
    reward(text, item)  -> float, the selection score of one train response
    check(answer, item) -> bool or float, the test score of one voted answer

``get_task(name)`` returns one; docs/tasks.md describes the data, prompts and scoring
of each. Some scorers are lenient (unanchored keyword patterns, a permissive MATH
matcher, GSM8K's last-number reward): the paper's numbers were measured with them.
"""

import ast
import hashlib
import json
import math
import os
import random
import re
import secrets
import subprocess
import sys
import urllib.request

# Countdown formulas are evaluated as eval() would, except that an integer power above
# this many bits is refused (9**9**9 would run for hours in uninterruptible C).
_MAX_POW_BITS = 1 << 20

DATA_DIR = os.environ.get(
    "LOGIT_PSV_DATA", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
)


def _arith_eval(expr):
    """eval(expr, {"__builtins__": None}, {}) for a +-*/ formula, minus integer power
    towers, which raise OverflowError (an invalid formula) instead of hanging."""
    tree = ast.parse(expr.lstrip(" \t"), mode="eval")  # eval() strips leading blanks too

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Tuple):  # only "()" is expressible in the charset
            return tuple(ev(element) for element in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = ev(node.operand)
            return +value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.FloorDiv):
                return a // b
            if isinstance(node.op, ast.Pow):
                if (
                    isinstance(a, int)
                    and isinstance(b, int)
                    and abs(a) > 1
                    and b * math.log2(abs(a)) > _MAX_POW_BITS
                ):
                    raise OverflowError("integer power too large")
                return a**b
        raise TypeError(f"unsupported formula element {type(node).__name__}")

    return ev(tree)


# --------------------------------------------------------------------------
# GSM8K
# --------------------------------------------------------------------------


class GSM8K:
    name = "gsm8k"
    max_tokens = 1024
    INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'

    def load(self, split, n=None):
        from datasets import load_dataset

        items = []
        for row in load_dataset("openai/gsm8k", "main", split=split):
            items.append(
                {
                    "messages": [
                        {"role": "user", "content": row["question"] + " " + self.INSTRUCTION}
                    ],
                    "ground_truth": row["answer"].split("#### ")[-1].replace(",", "").strip(),
                }
            )
            if n and len(items) >= n:
                break
        return items

    def _extract(self, text):
        """(strict "#### N" answer, flexible last-number answer) from the last 300 chars."""
        tail = text[-300:]

        def clean(answer):
            return answer.replace(",", "").replace("$", "")

        strict = re.findall(r"#### (\-?[0-9\.\,]+)", tail)
        nums = [a for a in re.findall(r"(\-?[0-9\.\,]+)", tail) if a not in ("", ".")]
        return (clean(strict[-1]) if strict else None, clean(nums[-1]) if nums else None)

    def extract(self, text, item=None):
        strict, flexible = self._extract(text)
        return strict if strict is not None else (flexible or "")

    def reward(self, text, item):
        # A wrong "#### N" still earns the reward when the last number is right.
        strict, flexible = self._extract(text)
        gt = item["ground_truth"]
        return float(strict == gt or flexible == gt)

    def check(self, answer, item):
        return answer == item["ground_truth"]


# --------------------------------------------------------------------------
# MATH-500: one 500-problem file, the first 200 rows select, the other 300 test.
# --------------------------------------------------------------------------


class MATH500:
    name = "math500"
    max_tokens = 2048
    split_at = 200
    INSTRUCTION = "Let's think step by step and output the final answer after ####"

    def load(self, split, n=None):
        from datasets import load_dataset

        rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
        rows = rows[: self.split_at] if split == "train" else rows[self.split_at :]
        return [
            {
                "messages": [{"role": "user", "content": r["problem"] + "\n\n" + self.INSTRUCTION}],
                "ground_truth": r["answer"],
            }
            for r in rows[:n]
        ]

    def extract(self, text, item=None):
        tail = text[-500:]
        # The last balanced \boxed{...} (despite the prompt asking for ####).
        boxed = []
        i = 0
        while (i := tail.find("\\boxed{", i)) != -1:
            depth, j = 1, i + 7
            while j < len(tail) and depth:
                depth += {"{": 1, "}": -1}.get(tail[j], 0)
                j += 1
            if depth == 0:
                boxed.append(tail[i + 7 : j - 1])
            i = j
        if boxed:
            return boxed[-1]
        for pat in (r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\\boxed\{([^}]+)\}"):
            m = re.findall(pat, tail)  # unbalanced-brace fallbacks
            if m:
                return m[-1]
        # Flexible fallbacks. Keywords are unanchored: "so" also fires inside "solution".
        for pat, flags in (
            (r"is\s+\\\(([^)]+)\\\)\.?\s*$", re.IGNORECASE | re.MULTILINE),
            (r"is\s+\$([^$]+)\$\.?\s*$", re.IGNORECASE | re.MULTILINE),
            (r"=\s*\\\(([^)]+)\\\)\.?\s*$", re.IGNORECASE | re.MULTILINE),
            (r"=\s*\$([^$]+)\$\.?\s*$", re.IGNORECASE | re.MULTILINE),
            (r"answer\s+is\s*[:\s]*\**\\\(([^)]+)\\\)\**", re.IGNORECASE),
            (r"answer\s+is\s*[:\s]*\**\$([^$]+)\$\**", re.IGNORECASE),
            (r"answer\s+is\s*[:\s]*\**([^\n.*]+)\**", re.IGNORECASE),
            (
                r"(?:therefore|thus|hence|so)\s*,?\s*(?:the\s+)?(?:answer\s+is\s*)?[:\s]*\**([^\n.*]+)\**",
                re.IGNORECASE,
            ),
            (r"(?:final\s+)?answer[:\s]+\**([^\n.*]+)\**", re.IGNORECASE),
            (r"=\s*([^\n=]+)$", re.IGNORECASE),
        ):
            m = re.findall(pat, tail, flags)
            if m:
                return m[-1].strip()
        return ""

    def reward(self, text, item):
        answer = self.extract(text)
        return float(bool(answer) and self._equiv(answer, item["ground_truth"]))

    def check(self, answer, item):
        return bool(answer) and self._equiv(answer, item["ground_truth"])

    @staticmethod
    def _simplify(s):
        for a, b in (
            ("\\left", ""),
            ("\\right", ""),
            ("\\,", ""),
            ("\\!", ""),
            ("\\;", ""),
            ("\\:", ""),
            ("\\ ", " "),
            ("\\quad", " "),
            ("\\qquad", " "),
            ("\\text{", "{"),
            ("\\mathrm{", "{"),
            ("\\mathbf{", "{"),
            ("\\textbf{", "{"),
            ("\\dfrac", "\\frac"),
            ("\\tfrac", "\\frac"),
            ("^{\\circ}", "^\\circ"),
            ("°", "^\\circ"),
        ):
            s = s.replace(a, b)
        for _ in range(3):  # thousand separators: 1,234,567 -> 1234567
            s = re.sub(r"(\d),(\d{3})", r"\1\2", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def _equiv(cls, ans, gt):
        """Normalized string equality, then a numeric comparison whose \\frac{p}{q} parse
        is a prefix match, then a comparison of the bare lists of numbers. The last two
        are lenient: they accept sign errors, 3 for 3\\pi, [3,5] for (3,5), ..."""
        ans, gt = (re.sub(r"\s+", " ", x.strip()) for x in (ans, gt))
        a, g = cls._simplify(ans), cls._simplify(gt)
        if (
            ans == gt
            or ans.replace(" ", "") == gt.replace(" ", "")
            or a == g
            or a.replace(" ", "") == g.replace(" ", "")
        ):
            return True
        # remove all spaces; lowercase pure-text answers (e.g. names)
        fa, fg = (x.replace(" ", "") for x in (a, g))
        if not re.search(r"[\\{}^_]", fa):
            fa = fa.lower()
        if not re.search(r"[\\{}^_]", fg):
            fg = fg.lower()
        if fa == fg:
            return True
        try:  # numeric: plain floats or \frac{a}{b}

            def val(s):
                m = re.match(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s.strip())
                return float(m.group(1)) / float(m.group(2)) if m else float(s)

            if abs(val(a) - val(g)) < 1e-9:
                return True
        except (ValueError, ZeroDivisionError):
            pass
        na, ng = (re.findall(r"-?\d+\.?\d*", x.replace(" ", "")) for x in (a, g))
        return (
            bool(na)
            and len(na) == len(ng)
            and all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(na, ng))
        )


# --------------------------------------------------------------------------
# Countdown: one 2200-puzzle file, the first 200 select, the other 2000 test.
# --------------------------------------------------------------------------


class Countdown:
    name = "countdown"
    max_tokens = 1024
    split_at = 200
    # Pinned to a commit: the repository's main branch has since changed this file.
    DATA_URL = (
        "https://raw.githubusercontent.com/VsonicV/es-at-scale/"
        "1775c7286b9123b8764c4ee3301ccd9dfe853f7e/countdown/data/countdown.json"
    )
    DATA_SHA256 = "f1b5c5865d9f4bc517991740175f14f6869a2f8e4b00f8602f4655fdc6ccd8ba"
    SYSTEM = (
        "You are a helpful assistant. You first think about the reasoning process "
        "in your mind and then provide the user with the answer."
    )
    TEMPLATE = (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. "
        "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    )

    def _rows(self):
        path = os.path.join(DATA_DIR, "countdown.json")
        if not os.path.exists(path):
            os.makedirs(DATA_DIR, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            urllib.request.urlretrieve(self.DATA_URL, tmp)
            os.replace(tmp, path)
        with open(path, "rb") as f:
            raw = f.read()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.DATA_SHA256:
            raise ValueError(f"{path}: sha256 {digest}, expected {self.DATA_SHA256}")
        return json.loads(raw)

    def load(self, split, n=None):
        rows = self._rows()
        rows = rows[: self.split_at] if split == "train" else rows[self.split_at :]
        return [
            {
                "messages": [
                    {"role": "system", "content": self.SYSTEM},
                    {
                        "role": "user",
                        "content": self.TEMPLATE.format(numbers=r["numbers"], target=r["target"]),
                    },
                ],
                "numbers": r["numbers"],
                "ground_truth": float(r["target"]),
            }
            for r in rows[:n]
        ]

    def _formula(self, text, numbers):
        """The last <answer> formula, if it uses exactly the given numbers."""
        tags = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
        formula = tags[-1].strip() if tags else ""
        if not formula or not re.match(r"^[0-9+\-*/()= ]+$", formula):
            return None
        if "=" in formula:  # keep the equation side that uses the given numbers
            sides = [s.strip() for s in formula.split("=") if s.strip()]
            sides = [
                s
                for s in sides
                if re.match(r"^[0-9+\-*/() ]+$", s)
                and sorted(int(x) for x in re.findall(r"\d+", s)) == sorted(numbers)
            ]
            if len(sides) != 1:
                return None
            formula = sides[0]
        used = [int(x) for x in re.findall(r"\d+", formula)]
        if any(x > 1000 for x in used) or sorted(used) != sorted(numbers):
            return None
        return formula

    def reward(self, text, item):
        """0.1 * format adherence + 1.0 * correct answer (max 1.1)."""
        wrapped = "<think>" + text  # the response continues an opened <think> block
        fmt = (
            1.0
            if re.match(r"^<think>.*?</think>\n<answer>.*?</answer>$", wrapped, re.DOTALL)
            else 0.1 * bool(re.search(r"<think>.*?</think>", wrapped, re.DOTALL))
            + 0.5 * bool(re.search(r"<answer>.*?</answer>", text, re.DOTALL))
        )
        answer = self.extract(text, item)
        return 0.1 * fmt + float(bool(answer) and self.check(answer, item))

    def extract(self, text, item):
        """The value of the formula, so that '1+2' and '2+1' pool their votes."""
        formula = self._formula(text, item["numbers"])
        if formula is None:
            return ""
        try:
            result = _arith_eval(formula)
            return str(int(round(result))) if abs(result - round(result)) < 1e-9 else str(result)
        except Exception:  # degenerate formulas can raise ZeroDivisionError, MemoryError, ...
            return ""

    def check(self, answer, item):
        try:
            return abs(float(answer) - item["ground_truth"]) < 1e-5
        except ValueError:
            return False


# --------------------------------------------------------------------------
# OlympiadBench (OE_TO_maths_en_COMP, 674 problems: 200 select, 474 test). Same
# scorer as MATH-500; only the data and the instruction differ.
# --------------------------------------------------------------------------


class OlympiadBench(MATH500):
    name = "olympiadbench"
    max_tokens = 2048
    split_at = 200
    CONFIG = "OE_TO_maths_en_COMP"
    INSTRUCTION = "Let's think step by step and output the final answer after ####."

    @staticmethod
    def _clean_gt(answer):
        # The answer comes as '$\\frac{1}{2}$'; boxed model answers have no $.
        answer = str(answer).strip()
        if answer.startswith("$"):
            answer = answer[1:]
        if answer.endswith("$"):
            answer = answer[:-1]
        return answer.strip()

    def load(self, split, n=None):
        from datasets import load_dataset

        rows = list(load_dataset("Hothan/OlympiadBench", self.CONFIG, split="train"))
        random.Random(0).shuffle(rows)  # the file is sorted by id
        rows = rows[: self.split_at] if split == "train" else rows[self.split_at :]
        items = []
        for r in rows[:n]:
            fa = r["final_answer"]
            if isinstance(fa, str):  # the hub parquet stores the list as its repr
                try:
                    fa = ast.literal_eval(fa)
                except (ValueError, SyntaxError):
                    fa = [fa]
            items.append(
                {
                    "messages": [
                        {"role": "user", "content": r["question"] + "\n\n" + self.INSTRUCTION}
                    ],
                    "ground_truth": self._clean_gt(fa[0] if fa else ""),
                }
            )
        return items


# --------------------------------------------------------------------------
# MBPP: all 964 rows (train + validation + test, in order), 200 select, 764 test.
# A member votes with its raw code string; a vote is correct if it passes every test.
# --------------------------------------------------------------------------

# Runs one candidate in a fresh interpreter (see MBPP._passes). Caps memory and CPU,
# execs setup + code then every test in one namespace, and prints the parent's
# one-time token only after all of them ran clean, so neither sys.exit() nor
# os._exit(0) from generated code can fake a pass.
_MBPP_HARNESS = r"""
import json, os, resource, sys
write, exit_ = os.write, os._exit
d = json.loads(sys.stdin.read())
for res, lim in ((resource.RLIMIT_AS, d["mem_bytes"]), (resource.RLIMIT_CPU, d["cpu_sec"])):
    try:
        resource.setrlimit(res, (lim, lim))
    except (ValueError, OSError):
        pass
token = d.pop("token").encode()
ns = {"__name__": "__candidate__"}
try:
    for src in [d["full"]] + d["tests"]:
        exec(src, ns)
except BaseException:
    exit_(1)
write(1, b"\n" + token + b"\n")
exit_(0)
"""


class MBPP:
    name = "mbpp"
    max_tokens = 2048
    split_at = 200
    SYSTEM = (
        "You are a Python programming assistant. Write clean, correct Python "
        "code to solve the given problem."
    )
    TEMPLATE = (
        "{text}\n\n"
        "Your code should pass these tests:\n{tests}\n\n"
        "Think through your solution in <think> </think> tags.\n"
        "Return your final Python code in <answer> </answer> tags, e.g.:\n"
        "<answer>\ndef solution(x):\n    return x + 1\n</answer>"
    )

    def load(self, split, n=None):
        from datasets import load_dataset

        ds = load_dataset("google-research-datasets/mbpp", "full")
        rows = []
        for s in ("train", "validation", "test"):  # 374 + 90 + 500 = 964, in order
            rows.extend(ds[s])
        rows = rows[: self.split_at] if split == "train" else rows[self.split_at :]
        items = []
        for it in rows[:n]:
            tests = list(it["test_list"])
            items.append(
                {
                    "messages": [
                        {"role": "system", "content": self.SYSTEM},
                        {
                            "role": "user",
                            "content": self.TEMPLATE.format(
                                text=it["text"], tests="\n".join(tests[:3])
                            ),
                        },
                    ],
                    "ground_truth": {
                        "test_list": tests,
                        "test_setup_code": it.get("test_setup_code", "") or "",
                    },
                }
            )
        return items

    @staticmethod
    def _code(text):
        for pat in (r"<answer>(.*?)</answer>", r"```python\n?(.*?)```", r"```\n?(.*?)```"):
            m = re.findall(pat, text, re.DOTALL)
            if m:
                return m[-1].strip()
        return ""

    @staticmethod
    def _passes(code, gt, timeout_sec=10, mem_mb=4096):
        """Run candidate code and its tests in an isolated subprocess (never fork: the
        vLLM process holds CUDA state and threads). A memory bomb hits RLIMIT_AS, a CPU
        spin RLIMIT_CPU, a C-level hang the wall-clock timeout; all count as failures.
        This is a resource limit, not a security sandbox."""
        if not code or not code.strip():
            return False
        setup = gt.get("test_setup_code", "") or ""
        full = (setup + "\n" if setup.strip() else "") + code + "\n"
        tests = list(gt.get("test_list", []))
        token = secrets.token_hex(16)
        payload = json.dumps(
            {
                "full": full,
                "tests": tests,
                "mem_bytes": int(mem_mb) * 1024 * 1024,
                "cpu_sec": int(timeout_sec),
                "token": token,
            }
        )
        try:
            r = subprocess.run(
                [sys.executable, "-c", _MBPP_HARNESS],
                input=payload,
                text=True,
                errors="replace",
                capture_output=True,
                timeout=timeout_sec + 5,
            )
            return r.returncode == 0 and token in r.stdout
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def extract(self, text, item=None):
        return self._code(text)

    def reward(self, text, item):
        return float(self._passes(self._code(text), item["ground_truth"]))

    def check(self, answer, item):
        return self._passes(answer, item["ground_truth"])


# --------------------------------------------------------------------------
# ROCStories: order 5 shuffled sentences. Votes are orders such as "BCEDA"; reward and
# accuracy are a lenient score, 0.6 * position + 0.4 * adjacency.
# --------------------------------------------------------------------------


class ROCStories:
    name = "rocstories"
    max_tokens = 64
    split_at = 200
    LABELS = ["A", "B", "C", "D", "E"]
    SYSTEM = (
        "You are a helpful assistant that excels at story comprehension and "
        "logical reasoning. Given shuffled sentences from a story, you carefully "
        "analyze the narrative flow and temporal cues to determine the correct "
        "chronological order."
    )
    ORDER_PATTERNS = (
        r"([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\s*,\s*([A-E])\s*,\s*([A-E])",
        r"([A-E])\s+([A-E])\s+([A-E])\s+([A-E])\s+([A-E])",
        r"([A-E])([A-E])([A-E])([A-E])([A-E])",
    )

    def load(self, split, n=None):
        from datasets import load_dataset

        rows = list(load_dataset("shawon/rocstories-combined", split="train"))
        # file order: the test set is every story after the selection ones
        rows = rows[: self.split_at] if split == "train" else rows[self.split_at :]
        items = []
        for r in rows[:n]:
            sents = "\n".join(
                f"Sentence {self.LABELS[i]}: {s}" for i, s in enumerate(r["shuffled_sentences"])
            )
            gold_labels = [self.LABELS[g] for g in r["gold_order"]]
            prompt = (
                f"Below are 5 sentences from a story, but they are in the wrong order.\n"
                f"Please arrange them in the correct chronological order.\n\n"
                f"Title: {r.get('title', '')}\n\n{sents}\n\n"
                f"Output the correct order as comma-separated letters (e.g., B,A,D,E,C).\n"
                f"Only output the letters, nothing else."
            )
            items.append(
                {
                    "messages": [
                        {"role": "system", "content": self.SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "ground_truth": {"gold_labels": gold_labels, "gold": "".join(gold_labels)},
                }
            )
        return items

    def extract(self, text, item=None):
        t = text.upper()
        for pat in self.ORDER_PATTERNS:
            m = re.search(pat, t)
            if m:
                labels = list(m.groups())
                break
        else:
            letters = re.findall(r"[A-E]", t)
            labels = letters[:5] if len(letters) >= 5 else []
        if len(labels) != 5 or set(labels) != set("ABCDE"):
            return ""  # not a permutation: abstain
        return "".join(labels)

    @staticmethod
    def _lenient(pred, gold_labels):
        if len(pred) != 5 or set(pred) != set("ABCDE"):
            return 0.0
        pos = sum(p == g for p, g in zip(pred, gold_labels)) / 5.0
        rank = {lab: i for i, lab in enumerate(pred)}
        adj = sum(rank[gold_labels[i]] < rank[gold_labels[i + 1]] for i in range(4)) / 4.0
        return 0.6 * pos + 0.4 * adj

    def reward(self, text, item):
        return self._lenient(self.extract(text), item["ground_truth"]["gold_labels"])

    def check(self, answer, item):
        return self._lenient(answer, item["ground_truth"]["gold_labels"])


# --------------------------------------------------------------------------
# USPTO-50K reaction classification (10 classes): the first 200 rows of the train
# split select, the 1001-row validation split is the test set.
# --------------------------------------------------------------------------


class USPTO:
    name = "uspto"
    DATASET_ID = "pingzhili/uspto-50k"
    DATASET_REVISION = "8296db0b791758743cb8efb5d62cc10d03c2b988"
    max_tokens = 1024
    SYSTEM = (
        "You are an expert organic chemist. Your task is to classify chemical reactions "
        "into one of 10 standard reaction categories based on the transformation type.\n\n"
        "## Reaction Classes:\n"
        "1: Heteroatom alkylation/arylation - N, O, S attacking C (e.g., SN2, ether formation)\n"
        "2: Acylation - Forming C=O bonds with N, O, S (e.g., amide, ester formation)\n"
        "3: C-C bond formation - New C-C bonds (e.g., Suzuki, Heck, Grignard)\n"
        "4: Heterocycle formation - Creating rings with N, O, S\n"
        "5: Protections - Adding protecting groups (Boc, Bn, TBS, etc.)\n"
        "6: Deprotections - Removing protecting groups\n"
        "7: Reductions - Adding H, removing O (e.g., ketone→alcohol, nitro→amine)\n"
        "8: Oxidations - Adding O, removing H (e.g., alcohol→ketone)\n"
        "9: Functional group interconversion - Changing one FG to another\n"
        "10: Functional group addition - Adding new FG to molecule (e.g., halogenation)"
    )
    TEMPLATE = (
        "Classify this reaction:\n\n"
        "Reactants >> Product:\n{rxn}\n\n"
        "Analyze the key transformation and output the class number (1-10) in <answer>X</answer> tags."
    )

    def load(self, split, n=None):
        from datasets import load_dataset

        hf_split = "train" if split == "train" else "validation"
        try:
            ds = load_dataset(self.DATASET_ID, revision=self.DATASET_REVISION, split=hf_split)
            rows = (ds[i] for i in range(len(ds) if n is None else min(len(ds), n)))
        except (ConnectionError, OSError):
            # No network access: read the pinned parquet from the Hub cache directly.
            import glob

            import pyarrow.parquet as pq
            from huggingface_hub import snapshot_download

            snapshot = snapshot_download(
                self.DATASET_ID,
                repo_type="dataset",
                revision=self.DATASET_REVISION,
                local_files_only=True,
            )
            parquet = sorted(glob.glob(os.path.join(snapshot, "data", f"{hf_split}-*.parquet")))
            if not parquet:
                raise FileNotFoundError(f"pinned USPTO parquet {hf_split!r} not in {snapshot}")
            table = pq.read_table(parquet)
            if n is not None:
                table = table.slice(0, min(n, table.num_rows))
            rows = iter(table.to_pylist())
        return [
            {
                "messages": [
                    {"role": "system", "content": self.SYSTEM},
                    {
                        "role": "user",
                        # drop the atom-map numbers
                        "content": self.TEMPLATE.format(
                            rxn=re.sub(r":(\d+)\]", "]", row["rxn_smiles"])
                        ),
                    },
                ],
                "ground_truth": str(row["class"]),
            }
            for row in rows
        ]

    def extract(self, text, item=None):
        m = re.findall(r"<answer>\s*(10|[1-9])\s*</answer>", text, re.IGNORECASE)
        if not m:
            m = re.findall(r"\b([1-9]|10)\b", text)  # else the last bare 1..10
        return m[-1] if m else ""

    def reward(self, text, item):
        return float(self.check(self.extract(text), item))

    def check(self, answer, item):
        try:
            return int(answer) == int(item["ground_truth"])
        except (ValueError, TypeError):
            return False


TASKS = {t.name: t for t in (Countdown, GSM8K, MATH500, OlympiadBench, MBPP, ROCStories, USPTO)}


def get_task(name, train_samples=200):
    """A task instance. ``train_samples`` is also where single-file datasets (all but
    GSM8K and USPTO) split their selection rows from their test rows."""
    task = TASKS[name]()
    if hasattr(task, "split_at"):
        task.split_at = train_samples
    return task
