"""Judge prompt variants."""

from __future__ import annotations

from .data import Pair

VARIANTS = ("baseline", "rubric", "rubric_fewshot")

_LABEL_TAG = {"A": "[[A]]", "B": "[[B]]", "tie": "[[TIE]]"}

BASELINE_SYSTEM = (
    "You are comparing two AI assistant answers to the same user question. "
    "Which answer is better? Reply with only your verdict: [[A]] if Assistant A is better, "
    "[[B]] if Assistant B is better, or [[TIE]] if they are equally good."
)

RUBRIC_SYSTEM = """You are an impartial judge comparing two AI assistant answers to the same user question.

Evaluate both answers on these criteria:
1. Helpfulness: does it address what the user actually needs?
2. Correctness: are the facts, reasoning, code, and math right? Errors weigh heavily.
3. Depth: appropriate level of detail and insight, without padding.
4. Instruction-following: does it respect every constraint in the question (format, length, style, scope)?

Answer order and answer length are not criteria. Do not favor an answer because it appears first or because it is longer.

Write brief reasoning (2-4 sentences) comparing the answers on these criteria. Then give your verdict on the final line, exactly one of: [[A]] if Assistant A is better, [[B]] if Assistant B is better, [[TIE]] if they are equally good."""


def format_pair(question: str, answer_a: str, answer_b: str) -> str:
    return (
        f"[User Question]\n{question}\n\n"
        f"[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n"
        f"[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
    )


def fewshot_system(anchors: list[Pair]) -> str:
    parts = [RUBRIC_SYSTEM, "\nFor calibration, here are example comparisons with verdicts from expert human judges:"]
    for i, p in enumerate(anchors, 1):
        parts.append(f"\n### Example {i}\n{format_pair(p.question, p.answer_a, p.answer_b)}\n\nExpert verdict: {_LABEL_TAG[p.human]}")
    parts.append("\nNow judge the next comparison the same way: brief reasoning, then the verdict on the final line.")
    return "\n".join(parts)


def build_messages(variant: str, question: str, answer_a: str, answer_b: str, anchors: list[Pair] | None = None) -> list[dict]:
    if variant == "baseline":
        system = BASELINE_SYSTEM
    elif variant == "rubric":
        system = RUBRIC_SYSTEM
    elif variant == "rubric_fewshot":
        if not anchors:
            raise ValueError("rubric_fewshot needs anchor examples")
        system = fewshot_system(anchors)
    else:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": format_pair(question, answer_a, answer_b)},
    ]
