"""Load MT-Bench human judgments, aggregate votes per pair, and split by question."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

DATASET = "lmsys/mt_bench_human_judgments"
SPLIT = "human"
LABELS = ("A", "B", "tie")

_WINNER = {"model_a": "A", "model_b": "B", "tie": "tie"}
SWAP = {"A": "B", "B": "A", "tie": "tie"}


@dataclass
class Pair:
    question_id: int
    model_a: str
    model_b: str
    question: str
    answer_a: str
    answer_b: str
    votes: dict[str, int] = field(default_factory=dict)
    human: str | None = None  # majority verdict: "A", "B", "tie", or None if no majority

    @property
    def key(self) -> str:
        return f"{self.question_id}|{self.model_a}|{self.model_b}"

    @property
    def n_votes(self) -> int:
        return sum(self.votes.values())

    @property
    def unanimous(self) -> bool:
        return len(self.votes) == 1


def load_rows() -> list[dict]:
    """Turn-1 rows of the human split."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split=SPLIT)
    return [r for r in ds if r["turn"] == 1]


def majority(votes: Counter) -> str | None:
    """Strict plurality; None when the top two labels have equal counts."""
    ranked = votes.most_common()
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def aggregate(rows: list[dict]) -> tuple[list[Pair], dict]:
    """Group raw judgments into one Pair per (question_id, model_a, model_b).

    The dataset contains the same pair in both orders, so model names are put in
    sorted order (flipping the winner accordingly) before votes are pooled.
    """
    pairs: dict[tuple, Pair] = {}
    counts: dict[tuple, Counter] = {}
    skipped = 0
    for r in rows:
        w = _WINNER.get(r["winner"])
        if w is None:
            skipped += 1
            continue
        a, b = r["model_a"], r["model_b"]
        conv_a, conv_b = r["conversation_a"], r["conversation_b"]
        if a > b:
            a, b, w, conv_a, conv_b = b, a, SWAP[w], conv_b, conv_a
        key = (r["question_id"], a, b)
        if key not in pairs:
            pairs[key] = Pair(
                question_id=r["question_id"],
                model_a=a,
                model_b=b,
                question=conv_a[0]["content"],
                answer_a=conv_a[1]["content"],
                answer_b=conv_b[1]["content"],
            )
            counts[key] = Counter()
        counts[key][w] += 1

    out, no_majority = [], 0
    for key in sorted(pairs):
        p = pairs[key]
        p.votes = dict(counts[key])
        p.human = majority(counts[key])
        if p.human is None:
            no_majority += 1
            continue
        out.append(p)
    stats = {
        "rows": len(rows),
        "rows_skipped_unknown_winner": skipped,
        "unique_pairs": len(pairs),
        "pairs_no_majority_dropped": no_majority,
        "pairs_with_majority": len(out),
    }
    return out, stats


def split_by_question(pairs: list[Pair], calib_questions: int, seed: int) -> tuple[list[Pair], list[Pair]]:
    """Disjoint calibration/eval split on question_id."""
    qids = sorted({p.question_id for p in pairs})
    rng = random.Random(seed)
    rng.shuffle(qids)
    calib_ids = set(qids[:calib_questions])
    calib = [p for p in pairs if p.question_id in calib_ids]
    evals = [p for p in pairs if p.question_id not in calib_ids]
    return calib, evals


def sample(pairs: list[Pair], n: int, seed: int) -> list[Pair]:
    if n >= len(pairs):
        return list(pairs)
    picked = random.Random(seed).sample(pairs, n)
    return sorted(picked, key=lambda p: p.key)


def select_anchors(calib: list[Pair], seed: int, include_tie: bool = True, k: int = 3, max_chars: int = 1500) -> list[Pair]:
    """Pick k few-shot anchors from the calibration split.

    Prefers unanimous, multi-vote pairs with answers short enough to keep the
    prompt small, and covers each verdict label before repeating one.
    """
    pool = [p for p in calib if p.unanimous and len(p.answer_a) <= max_chars and len(p.answer_b) <= max_chars]
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool.sort(key=lambda p: -p.n_votes)  # stable: random order within same vote count
    wanted = ["A", "B", "tie"] if include_tie else ["A", "B"]
    anchors: list[Pair] = []
    i = 0
    while len(anchors) < k:
        label = wanted[i % len(wanted)]
        i += 1
        choice = next((p for p in pool if p.human == label and p not in anchors), None)
        if choice is not None:
            anchors.append(choice)
        elif i > 3 * k:
            break
    return anchors


def prepare(rows: list[dict], n: int, seed: int, calib_questions: int, exclude_ties: bool):
    """Aggregate, split, filter, sample. Returns (calib, eval_sample, stats)."""
    pairs, stats = aggregate(rows)
    calib, evals = split_by_question(pairs, calib_questions, seed)
    if exclude_ties:
        calib = [p for p in calib if p.human != "tie"]
        evals = [p for p in evals if p.human != "tie"]
    eval_sample = sample(evals, n, seed)
    stats.update(
        calib_questions=len({p.question_id for p in calib}),
        calib_pairs=len(calib),
        eval_questions=len({p.question_id for p in evals}),
        eval_pairs_available=len(evals),
        eval_pairs_sampled=len(eval_sample),
        eval_label_counts=dict(Counter(p.human for p in eval_sample)),
    )
    return calib, eval_sample, stats
