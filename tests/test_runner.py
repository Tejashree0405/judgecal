import json
import re
from types import SimpleNamespace

from judgecal import report
from judgecal.cli import main
from judgecal.llm import Cache, LLMClient
from judgecal.prompts import build_messages
from judgecal.runner import RunConfig, run


def longer_answer_judge(**kwargs):
    """Fake judge that always prefers the longer answer; replies 'unsure' for one model."""
    user = kwargs["messages"][-1]["content"]
    if kwargs["model"] == "fake/broken":
        text = "unsure"
    elif "Reply with exactly" in user:
        text = "[[A]]"
    else:
        a = re.search(r"Assistant A's Answer\]\n(.*?)\n\[The End", user, re.S).group(1)
        b = re.search(r"Assistant B's Answer\]\n(.*?)\n\[The End", user, re.S).group(1)
        text = "Reasoning.\n[[A]]" if len(a) >= len(b) else "Reasoning.\n[[B]]"
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_end_to_end_run_and_report(tmp_path, synthetic_rows):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return longer_answer_judge(**kw)

    client = LLMClient(Cache(tmp_path / "c.sqlite"), completion_fn=fake)
    cfg = RunConfig(models=["fake/good", "fake/broken"], n=10, seed=0, calib_questions=4, concurrency=3, out_dir=str(tmp_path / "out"))
    res = run(cfg, client, rows=synthetic_rows)

    assert len(res["records"]) == 2 * 3 * 10 * 2
    by = {(m["model"], m["variant"]): m for m in res["metrics"]}
    good = by[("fake/good", "baseline")]
    assert good["position_flip_rate"] == 0.0  # length-based judge ignores order
    assert good["length_bias"] == 1.0
    broken = by[("fake/broken", "rubric")]
    assert broken["n_unparseable"] == broken["n_scored"] == 20
    assert broken["accuracy"] == 0.0

    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    md = (tmp_path / "out" / "report.md").read_text()
    assert saved["metrics"] == res["metrics"]
    assert "## Side by side" in md and "`fake/broken`" in md

    # Rerun is fully cached.
    n_before = len(calls)
    run(cfg, client, rows=synthetic_rows)
    assert len(calls) == n_before


def test_fewshot_prompt_has_no_eval_questions(tmp_path, synthetic_rows):
    seen = []

    def fake(**kw):
        seen.append(kw["messages"])
        return longer_answer_judge(**kw)

    client = LLMClient(Cache(tmp_path / "c.sqlite"), completion_fn=fake)
    cfg = RunConfig(models=["fake/good"], variants=["rubric_fewshot"], n=1000, calib_questions=4, out_dir=str(tmp_path / "o"))
    res = run(cfg, client, rows=synthetic_rows)
    eval_qids = {p["question_id"] for p in res["pairs"].values()}
    system = seen[-1][0]["content"]
    assert system.count("Expert verdict:") == 3
    for qid in eval_qids:
        assert f"question {qid}\n" not in system


def test_report_cli_merges_files(tmp_path, synthetic_rows):
    outs = []
    for model in ["fake/one", "fake/two"]:
        client = LLMClient(Cache(tmp_path / "c.sqlite"), completion_fn=longer_answer_judge)
        out = tmp_path / model.split("/")[1]
        run(RunConfig(models=[model], n=6, calib_questions=4, out_dir=str(out)), client, rows=synthetic_rows)
        outs.append(str(out / "results.json"))
    main(["report", *outs, "--out", str(tmp_path / "cmp.md")])
    md = (tmp_path / "cmp.md").read_text()
    assert "`fake/one`" in md and "`fake/two`" in md and "## Side by side" in md


def test_build_messages_rejects_unknown_variant():
    import pytest

    with pytest.raises(ValueError):
        build_messages("nope", "q", "a", "b")
