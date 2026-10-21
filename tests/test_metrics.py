import pytest

from judgecal.metrics import cohen_kappa, compute


def test_kappa_known_values():
    assert cohen_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == 1.0
    # po = 0.5, pe = 0.5 -> 0
    assert cohen_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]) == pytest.approx(0.0)
    # classic 2x2 example: po=0.7, pe=0.5 -> 0.4
    t = ["A"] * 5 + ["B"] * 5
    p = ["A"] * 4 + ["B"] + ["A"] * 2 + ["B"] * 3
    assert cohen_kappa(t, p) == pytest.approx(0.4)
    assert cohen_kappa([], []) is None


def rec(pair, swapped, response, error=None):
    return {"model": "m", "variant": "baseline", "pair": pair, "swapped": swapped, "response": response, "error": error}


PAIRS = {
    # p1: B is longer, human picked A (shorter)
    "p1": {"human": "A", "len_a": 10, "len_b": 100},
    # p2: A is longer, human picked A (longer)
    "p2": {"human": "A", "len_a": 100, "len_b": 10},
    # p3: B longer, human tie
    "p3": {"human": "tie", "len_a": 10, "len_b": 100},
}


def test_swap_mapping_and_position_flips():
    records = [
        rec("p1", False, "[[A]]"), rec("p1", True, "[[B]]"),   # consistent: original A both times
        rec("p2", False, "[[A]]"), rec("p2", True, "[[A]]"),   # flip: A first, then B in original frame
        rec("p3", False, "[[TIE]]"), rec("p3", True, "[[TIE]]"),
    ]
    m = compute(records, PAIRS)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["position_flip_rate"] == pytest.approx(1 / 3)
    assert m["position_flip_n"] == 3
    # p2 inconsistent -> tie, human A -> wrong
    assert m["accuracy_swap_consistent"] == pytest.approx(2 / 3)


def test_length_bias():
    records = [
        rec("p1", False, "[[B]]"), rec("p1", True, "[[B]]"),  # original verdicts: B (longer), A
        rec("p2", False, "[[A]]"), rec("p2", True, "[[B]]"),  # human picked longer: not in denominator
        rec("p3", False, "[[B]]"), rec("p3", True, "[[A]]"),  # picks longer B twice
    ]
    m = compute(records, PAIRS)
    assert m["length_bias_n"] == 4
    assert m["length_bias"] == pytest.approx(3 / 4)


def test_unparseable_counted_and_errors_excluded():
    records = [
        rec("p1", False, "no idea"), rec("p1", True, "[[B]]"),
        rec("p2", False, None, error="RateLimitError: quota"), rec("p2", True, "[[B]]"),
    ]
    m = compute(records, PAIRS)
    assert m["n_unparseable"] == 1
    assert m["n_errors"] == 1
    assert m["n_scored"] == 3
    assert m["unparseable_rate"] == pytest.approx(1 / 3)
    # unparseable counts as wrong: correct = p1 swapped (B->A) and p2 swapped (B->A)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert m["verdict_counts"]["invalid"] == 1
    assert m["position_flip_n"] == 0  # no pair has both orderings parsed
