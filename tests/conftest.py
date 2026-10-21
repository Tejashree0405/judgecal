import pytest


def make_row(qid, a, b, winner, judge="j1", ans_a=None, ans_b=None, turn=1):
    q = f"question {qid}"
    return {
        "question_id": qid,
        "model_a": a,
        "model_b": b,
        "winner": winner,
        "judge": judge,
        "turn": turn,
        "conversation_a": [{"role": "user", "content": q}, {"role": "assistant", "content": ans_a or f"{a} answer to {qid}"}],
        "conversation_b": [{"role": "user", "content": q}, {"role": "assistant", "content": ans_b or f"{b} answer to {qid}"}],
    }


@pytest.fixture
def synthetic_rows():
    """20 questions x 3 model pairs, with a mix of A/B/tie winners."""
    rows = []
    winners = ["model_a", "model_b", "tie"]
    for qid in range(1, 21):
        for i, (a, b) in enumerate([("m1", "m2"), ("m1", "m3"), ("m2", "m3")]):
            w = winners[(qid + i) % 3]
            rows.append(make_row(qid, a, b, w, ans_a="short", ans_b="a much longer answer " * (i + 1)))
    return rows
