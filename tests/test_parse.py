import pytest

from judgecal.parse import parse_verdict


@pytest.mark.parametrize(
    "text,expected",
    [
        ("[[A]]", "A"),
        ("[[B]]", "B"),
        ("[[TIE]]", "tie"),
        ("[[tie]]", "tie"),
        ("[[C]]", "tie"),
        ("[[ b ]]", "B"),
        ("**[[A]]**", "A"),
        ("Assistant A is wordy but [[A]] vs [[B]]... on reflection B is correct.\n[[B]]", "B"),
        ("Reasoning here.\nVerdict: A", "A"),
        ("Final verdict - Assistant B", "B"),
        ("**Verdict:** Tie", "tie"),
        ("My verdict is a tie.", "tie"),
        ("A", "A"),
        ("b.", "B"),
        ("Tie", "tie"),
    ],
)
def test_parses(text, expected):
    assert parse_verdict(text) == expected


@pytest.mark.parametrize("text", [None, "", "I cannot decide.", "Both are great answers.", "[[D]]", "Answer"])
def test_unparseable(text):
    assert parse_verdict(text) is None
