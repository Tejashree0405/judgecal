"""Run every (model, variant, pair, ordering) judge call and save results."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import data, metrics, report
from .llm import LLMClient, QuotaExhausted
from .prompts import VARIANTS, build_messages

DEFAULT_MODEL = "gemini/gemini-3.8-flash"
PREFLIGHT = [{"role": "user", "content": "Reply with exactly: [[A]]"}]


@dataclass
class RunConfig:
    models: list[str] = field(default_factory=lambda: [DEFAULT_MODEL])
    variants: list[str] = field(default_factory=lambda: list(VARIANTS))
    n: int = 200
    seed: int = 0
    exclude_ties: bool = False
    calib_questions: int = 8
    concurrency: int = 4
    out_dir: str = "results"
    # Stop early once this many calls in a row fail after all retries (e.g. daily quota hit).
    max_consecutive_failures: int = 5


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def preflight(client: LLMClient, models: list[str]) -> None:
    """One tiny call per model so a bad model name or key fails before the real run."""
    for model in models:
        try:
            res = client.complete(model, PREFLIGHT)
        except Exception as exc:
            raise SystemExit(f"Preflight call to {model} failed: {type(exc).__name__}: {exc}") from exc
        _log(f"preflight {model}: ok{' (cached)' if res.cached else ''} -> {res.text.strip()[:40]!r}")


def run(config: RunConfig, client: LLMClient, rows: list[dict] | None = None) -> dict:
    for v in config.variants:
        if v not in VARIANTS:
            raise SystemExit(f"Unknown variant {v!r}; choose from {', '.join(VARIANTS)}")

    rows = rows if rows is not None else data.load_rows()
    calib, eval_pairs, data_stats = data.prepare(
        rows, config.n, config.seed, config.calib_questions, config.exclude_ties
    )
    anchors = data.select_anchors(calib, config.seed, include_tie=not config.exclude_ties)
    if "rubric_fewshot" in config.variants and len(anchors) < 3:
        _log(f"warning: only {len(anchors)} anchor examples available in the calibration split")
    eval_qids = {p.question_id for p in eval_pairs}
    assert not eval_qids & {a.question_id for a in anchors}, "anchor leaked into eval split"

    preflight(client, config.models)

    jobs = [
        (model, variant, pair, swapped)
        for model in config.models
        for variant in config.variants
        for pair in eval_pairs
        for swapped in (False, True)
    ]
    _log(f"{len(eval_pairs)} eval pairs x {len(config.variants)} variants x {len(config.models)} models x 2 orders = {len(jobs)} calls")

    stop = threading.Event()
    lock = threading.Lock()
    state = {"consecutive_failures": 0, "done": 0}

    def work(job):
        model, variant, pair, swapped = job
        rec = {"model": model, "variant": variant, "pair": pair.key, "swapped": swapped, "response": None, "error": None}
        if stop.is_set():
            rec["error"] = "skipped: run stopped early"
            return rec
        a, b = (pair.answer_b, pair.answer_a) if swapped else (pair.answer_a, pair.answer_b)
        messages = build_messages(variant, pair.question, a, b, anchors)
        try:
            rec["response"] = client.complete(model, messages).text
            with lock:
                state["consecutive_failures"] = 0
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            with lock:
                state["consecutive_failures"] += 1
                if isinstance(exc, QuotaExhausted) and not stop.is_set():
                    stop.set()
                    _log(f"stopping: {exc}. Rerun after the quota resets; finished calls are cached.")
                elif state["consecutive_failures"] >= config.max_consecutive_failures and not stop.is_set():
                    stop.set()
                    _log(f"stopping early after {config.max_consecutive_failures} consecutive failures; last: {rec['error']}")
        return rec

    records = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        futures = [pool.submit(work, j) for j in jobs]
        for fut in as_completed(futures):
            records.append(fut.result())
            state["done"] += 1
            if state["done"] % 50 == 0 or state["done"] == len(jobs):
                _log(f"  {state['done']}/{len(jobs)} calls  {client.stats}")

    order = {(m, v, p.key, s): i for i, (m, v, p, s) in enumerate(jobs)}
    records.sort(key=lambda r: order[(r["model"], r["variant"], r["pair"], r["swapped"])])

    pair_info = {
        p.key: {
            "question_id": p.question_id,
            "model_a": p.model_a,
            "model_b": p.model_b,
            "human": p.human,
            "votes": p.votes,
            "len_a": len(p.answer_a),
            "len_b": len(p.answer_b),
        }
        for p in eval_pairs
    }
    results = {
        "config": asdict(config),
        "data": data_stats,
        "anchors": [{"key": a.key, "human": a.human} for a in anchors],
        "client_stats": dict(client.stats),
        "metrics": metrics.compute_all(records, pair_info),
        "pairs": pair_info,
        "records": records,
    }
    save(results, Path(config.out_dir))
    return results


def save(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "report.md").write_text(report.render([results]))
    _log(f"wrote {out_dir / 'results.json'} and {out_dir / 'report.md'}")
