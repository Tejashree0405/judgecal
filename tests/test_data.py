from collections import Counter

from judgecal import data
from tests.conftest import make_row


def test_aggregate_merges_reversed_pairs_and_flips_winner():
    rows = [
        make_row(1, "m1", "m2", "model_a", judge="j1"),
        make_row(1, "m2", "m1", "model_b", judge="j2"),  # same preference: m1 wins
        make_row(1, "m2", "m1", "model_a", judge="j3"),  # m2 wins
    ]
    pairs, stats = data.aggregate(rows)
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.model_a, p.model_b) == ("m1", "m2")
    assert p.votes == {"A": 2, "B": 1}
    assert p.human == "A"
    assert p.answer_a == "m1 answer to 1"  # answers follow the swapped model order
    assert stats["unique_pairs"] == 1


def test_majority_keeps_ties_and_drops_splits():
    assert data.majority(Counter({"tie": 2, "A": 1})) == "tie"
    assert data.majority(Counter({"A": 1, "B": 1})) is None
    rows = [make_row(1, "m1", "m2", "model_a", judge="j1"), make_row(1, "m1", "m2", "model_b", judge="j2")]
    pairs, stats = data.aggregate(rows)
    assert pairs == [] and stats["pairs_no_majority_dropped"] == 1


def test_split_is_disjoint_by_question(synthetic_rows):
    pairs, _ = data.aggregate(synthetic_rows)
    calib, evals = data.split_by_question(pairs, calib_questions=4, seed=1)
    cq, eq = {p.question_id for p in calib}, {p.question_id for p in evals}
    assert len(cq) == 4 and not cq & eq and len(cq | eq) == 20


def test_sampling_is_seeded(synthetic_rows):
    _, a, _ = data.prepare(synthetic_rows, n=10, seed=3, calib_questions=4, exclude_ties=False)
    _, b, _ = data.prepare(synthetic_rows, n=10, seed=3, calib_questions=4, exclude_ties=False)
    _, c, _ = data.prepare(synthetic_rows, n=10, seed=4, calib_questions=4, exclude_ties=False)
    assert [p.key for p in a] == [p.key for p in b]
    assert [p.key for p in a] != [p.key for p in c]
    assert len(a) == 10


def test_exclude_ties(synthetic_rows):
    calib, evals, stats = data.prepare(synthetic_rows, n=1000, seed=0, calib_questions=4, exclude_ties=True)
    assert all(p.human != "tie" for p in calib + evals)
    assert "tie" not in stats["eval_label_counts"]


def test_anchors_come_from_calibration_split_only(synthetic_rows):
    calib, evals, _ = data.prepare(synthetic_rows, n=1000, seed=0, calib_questions=4, exclude_ties=False)
    anchors = data.select_anchors(calib, seed=0)
    assert len(anchors) == 3
    assert {a.human for a in anchors} == {"A", "B", "tie"}
    assert not {a.question_id for a in anchors} & {p.question_id for p in evals}


def test_anchors_without_ties(synthetic_rows):
    calib, _, _ = data.prepare(synthetic_rows, n=1000, seed=0, calib_questions=4, exclude_ties=True)
    anchors = data.select_anchors(calib, seed=0, include_tie=False)
    assert len(anchors) == 3 and {a.human for a in anchors} == {"A", "B"}
