
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import re
import signal
from typing import Optional, List, Tuple
from math_verify import parse
from math_verify import verify as m_verify
from dataclasses import dataclass, field
@dataclass
class ScoringConfig:
    correct_score: float = 1.0
    incorrect_score: float = 0.0

scoring_config = ScoringConfig()


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None

def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


# Constants for normalization
SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Args:
        final_answer: The answer string to normalize

    Returns:
        Normalized answer string
    """
    final_answer = final_answer.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    # Extract and normalize LaTeX math
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    # Normalize numbers
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer.strip()


def is_correct_minerva(solution_str: str, gt: str, gt_need_extract: bool = False, answer_pattern: str = r"(?i)Answer\s*:\s*([^\n]+)") -> tuple[bool, str]:
    """Check if the solution is correct according to Minerva criteria.

    Args:
        solution_str: The solution string to check
        gt: The ground truth answer
        gt_need_extract: Whether the ground truth needs extraction
        answer_pattern: Regex pattern to extract the answer

    Returns:
        Tuple of (is_correct, normalized_prediction)
    """
    # Extract answer from solution
    match = re.findall(answer_pattern, solution_str)
    extracted_answer = match[-1] if match else "[INVALID]"
    pred = normalize_final_answer(extracted_answer)

    # Process ground truth
    if gt_need_extract:
        gt = normalize_final_answer(remove_boxed(last_boxed_only_string(gt)))
    else:
        gt = normalize_final_answer(gt)

    return (pred == gt), pred


def is_correct_strict_box(pred: str, gt: str, pause_tokens_index: Optional[list[int]] = None) -> tuple[int, Optional[str]]:
    """Check if the prediction is correct using strict boxed answer criteria.

    Args:
        pred: The prediction string
        gt: The ground truth answer
        pause_tokens_index: Indices of pause tokens

    Returns:
        Tuple of (score, extracted_prediction)
    """
    # Extract the relevant part of the prediction
    if pause_tokens_index is not None:
        assert len(pause_tokens_index) == 4
        pred = pred[pause_tokens_index[-1] - 100 :]
    else:
        pred = pred[-100:]

    # Extract and check the boxed answer
    boxed_pred = last_boxed_only_string(pred)

    # print(boxed_pred)

    extracted_pred = remove_boxed(boxed_pred) if boxed_pred is not None else None

    # print(extracted_pred, gt, extracted_pred == gt)

    return 1 if (extracted_pred == gt) else -1, extracted_pred


def verify(solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None) -> bool:
    """Verify if the solution is correct.

    Args:
        solution_str: The solution string to verify
        answer: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        True if the solution is correct, False otherwise
    """
    if strict_box_verify:
        correct, pred = is_correct_strict_box(solution_str, answer, pause_tokens_index)
        return correct == 1, pred

    correct, pred = is_correct_minerva(solution_str, answer)
    return correct, pred

def math_verify_from_sky(solution_str: str, ground_truth: str):
    # print(ground_truth)
    ground_truth = [ground_truth] if isinstance(ground_truth, str) else ground_truth
    try:
        math_verify_parsed = parse(solution_str, parsing_timeout=10)
    except Exception:
        return scoring_config.incorrect_score

    # print(math_verify_parsed)

    # print(f"pass 1: {math_verify_parsed}")

    # 0 if parsing is problematic
    if len(math_verify_parsed) < 2:
        return scoring_config.incorrect_score

    # print(f"pass 2: {math_verify_parsed}")

    # We perform a quick string match first
    if math_verify_parsed[1] in ground_truth:
        return scoring_config.correct_score
    
    # print(f"pass 3: {math_verify_parsed}")
    
    # We now fallback to semantic verification
    for gt in ground_truth:
        try:
            if m_verify(
                parse(f"\\boxed{{{gt}}}", parsing_timeout=10),
                math_verify_parsed,
                timeout_seconds=10,
            ):
                return scoring_config.correct_score
        except Exception:
            continue
    # print(f"pass 4: {math_verify_parsed}")
    
    # Very unlikely to be correct after the above matches
    return scoring_config.incorrect_score


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
    is_longcot: bool = False,
    is_use_math_verify: bool = False,
) -> float:

    # ✅ math-verify path: parse only a small tail for speed
    tail = solution_str[-500:]
    boxed = last_boxed_only_string(tail)
    if boxed is None:
        # fallback: try Minerva 'Answer:' pattern from the FULL string
        correct, _pred = is_correct_minerva(solution_str, ground_truth)
        return scoring_config.correct_score if correct else scoring_config.incorrect_score

    # Keep the boxed wrapper as expected by your parser
    # (math_verify_from_sky expects a string it can parse; you already pass boxed or full string)
    # Here we pass the boxed segment only:
    reward = math_verify_from_sky(boxed, ground_truth)
    return reward


### more test

CASES: List[Tuple[str, str, bool, str]] = [
    # --- Display wrappers: $$ ... $$ vs \[ ... \] ---
    (
        r"Here we go: $$ \boxed{\frac{1}{2}+32c} $$",
        r"\frac{1}{2}+32c",
        True,
        "display-$$ correct",
    ),
    (
        r"The answer is \[ \boxed{$\frac{1}{2}+32c$} \].",
        r"\frac{1}{2}+32c",
        True,
        r"display-\[\] correct",
    ),

    # --- Inline math still fine (boxed lives inside) ---
    (
        r"The result is $\boxed{7}$ by inspection.",
        r"7",
        True,
        "inline-$ correct",
    ),

    # --- Algebraic equivalence (math_verify should accept) ---
    (
        r"Finally: $$ \boxed{\frac{2}{4}} $$",
        r"\frac{1}{2}",
        True,
        "algebraically equivalent fraction",
    ),
    (
        r"Area: \[ \boxed{2x+2x} \]",
        r"4x",
        True,
        "algebraic simplification",
    ),
    (
        r"Expanded form: \[ \boxed{(x+1)^2} \]",
        r"x^2+2x+1",
        True,
        "polynomial equivalence",
    ),
    (
        r"Trig: \[ \boxed{\sin^2\theta + \cos^2\theta} \]",
        r"1",
        True,
        "trig identity",
    ),

    # --- Multiple boxed: last one should be used ---
    (
        r"Try 1: \boxed{123}. But the final is \boxed{$456$}.",
        r"456",
        True,
        "take-last-boxed",
    ),
    (
        r"Wrong first \boxed{5} then correct \boxed{7}.",
        r"7",
        True,
        "take-last-boxed-2",
    ),

    # --- Text wrappers, spaces, percent sign ---
    (
        r"The accuracy is \[ \boxed{\text{93\%}} \].",
        r"93\%",
        True,
        r"text-wrapped percent",
    ),
    (
        r"The accuracy is $$ \boxed{93\%} $$.",
        r"93\%",
        True,
        "raw percent",
    ),

    # --- Form variants of the same value ---
    (
        r"The slope is \boxed{1/2}.",
        r"\frac{1}{2}",
        True,
        "shorthand fraction",
    ),
    (
        r"The slope is \boxed{\frac12}.",
        r"\frac{1}{2}",
        True,
        r"\frac12 shorthand",
    ),
    (
        r"Thus, the statement is always true for odd perfect numbers \(q^k n^2\).\n"
        r"\[ \boxed{Yes} \]\n",
        r"yes",
        True,
        "boxed yes",
    ),
    # (
    #     r"Thus, the statement is always true for odd perfect numbers \(q^k n^2\).
    #     \[
    #     \boxed{Yes}
    #     \]
    #     ",
    #     True,
    #     "boxed yes",
    # )

    # --- Negatives (should be incorrect) ---
    (
        r"Answer: \boxed{8}",
        r"9",
        False,
        "wrong integer",
    ),
    (
        r"No box here, only $$ 42 $$.",
        r"42",
        False,
        "no boxed → should fail strict boxed extract",
    ),
    (
        r"Malformed: \boxed{(1+2}",
        r"3",
        False,
        "unbalanced boxed",
    ),

    # --- Edge: long preamble, answer at end, both wrappers present ---
    (
        "blah " * 200 + r" $$\frac{1}{2}+7c$$ and finally \[ \boxed{\frac{1}{2}+32c} \] ",
        r"\frac{1}{2}+32c",
        True,
        "long tail truncation still finds last boxed",
    ),
]

def run(is_use_math_verify: bool):
    print(f"\n=== Running tests (is_use_math_verify={is_use_math_verify}) ===")
    ok = 0
    for sol, gt, should_pass, name in CASES:
        try:
            score = compute_score(sol, gt, is_use_math_verify=is_use_math_verify)
            passed = (score > 0.5)  # assuming correct_score=1.0, incorrect<=0
        except Exception as e:
            passed = False
            score = f"EXC: {e}"
        status = "PASS" if passed == should_pass else "FAIL"
        print(f"[{status}] {name:30s} → score={score!r}")
        ok += (status == "PASS")
    print(f"\n{ok}/{len(CASES)} cases passed.")


# test the compute_score function
if __name__ == '__main__':
    solution_str = "Assistant: Here is the solution to the problem.\n $$\\frac{1}{2} + 7c$$\n The answer is \[ \\boxed{\\frac{1}{2} + 32c}. \]"
    ground_truth = "\\frac{1}{2} + 32c"
    print(compute_score(solution_str, ground_truth, is_use_math_verify=True)) #1.0
    ground_truth = "\\frac{1}{3}"
    print(compute_score(solution_str, ground_truth, is_use_math_verify=True)) #0.1

    solution_str = "The answer is \\boxed{1/2}"
    ground_truth = "\\frac{1}{2}"
    print(compute_score(solution_str, ground_truth, is_use_math_verify=True)) #0

    run(is_use_math_verify=False)
    run(is_use_math_verify=True)



