"""Agreement and bias metrics computed from per-call records.

Each record is one judge call on one ordering of a pair:
    {"model", "variant", "pair", "swapped", "response", "error"}
Verdicts are parsed here (not at run time), so parser fixes apply to cached runs.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from .data import SWAP
from .parse import parse_verdict

INVALID = "invalid"


def cohen_kappa(y_true: list[str], y_pred: list[str]) -> float | None:
    n = len(y_true)
    if n == 0:
        return None
    labels = set(y_true) | set(y_pred)
    po = sum(t == p for t, p in zip(y_true, y_pred)) / n
    ct, cp = Counter(y_true), Counter(y_pred)
    pe = sum(ct[label] * cp[label] for label in labels) / (n * n)
    if pe == 1:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def verdict_in_original_order(record: dict) -> str | None:
    v = parse_verdict(record.get("response"))
    if v is not None and record["swapped"]:
        v = SWAP[v]
    return v


def _longer(pair: dict) -> str | None:
    if pair["len_a"] > pair["len_b"]:
        return "A"
    if pair["len_b"] > pair["len_a"]:
        return "B"
    return None


def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def compute(records: list[dict], pairs: dict[str, dict]) -> dict:
    """Metrics for one (model, variant). `pairs` maps pair key -> {human, len_a, len_b}."""
    scored = [r for r in records if not r.get("error")]
    errors = len(records) - len(scored)

    y_true, y_pred = [], []
    by_pair: dict[str, dict[bool, str]] = defaultdict(dict)
    unparseable = 0
    length_den = length_num = 0
    for r in scored:
        pair = pairs[r["pair"]]
        v = verdict_in_original_order(r)
        if v is None:
            unparseable += 1
        pred = v or INVALID
        y_true.append(pair["human"])
        y_pred.append(pred)
        by_pair[r["pair"]][r["swapped"]] = pred

        longer = _longer(pair)
        if v is not None and longer is not None and pair["human"] != longer:
            length_den += 1
            length_num += v == longer

    # Position bias: both orderings parsed, verdicts disagree once mapped back.
    both = [(o[False], o[True]) for o in by_pair.values() if False in o and True in o]
    both_parsed = [(x, y) for x, y in both if INVALID not in (x, y)]
    flips = sum(x != y for x, y in both_parsed)

    # MT-Bench convention: a verdict that changes with order counts as a tie.
    consistent_true, consistent_pred = [], []
    for key, o in by_pair.items():
        if False in o and True in o:
            x, y = o[False], o[True]
            consistent_true.append(pairs[key]["human"])
            consistent_pred.append(INVALID if INVALID in (x, y) else (x if x == y else "tie"))

    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return {
        "n_pairs": len({r["pair"] for r in records}),
        "n_calls": len(records),
        "n_scored": len(scored),
        "n_errors": errors,
        "n_unparseable": unparseable,
        "unparseable_rate": _rate(unparseable, len(scored)),
        "accuracy": _rate(correct, len(scored)),
        "kappa": cohen_kappa(y_true, y_pred),
        "accuracy_swap_consistent": _rate(
            sum(t == p for t, p in zip(consistent_true, consistent_pred)), len(consistent_true)
        ),
        "kappa_swap_consistent": cohen_kappa(consistent_true, consistent_pred),
        "position_flip_rate": _rate(flips, len(both_parsed)),
        "position_flip_n": len(both_parsed),
        "length_bias": _rate(length_num, length_den),
        "length_bias_n": length_den,
        "verdict_counts": dict(Counter(y_pred)),
    }


def compute_all(records: list[dict], pairs: dict[str, dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["model"], r["variant"])].append(r)
    return [{"model": m, "variant": v, **compute(rs, pairs)} for (m, v), rs in groups.items()]
