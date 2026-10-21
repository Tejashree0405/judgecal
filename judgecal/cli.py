"""Command-line interface: `judgecal run` and `judgecal report`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .prompts import VARIANTS
from .runner import DEFAULT_MODEL, RunConfig


def _cmd_run(args: argparse.Namespace) -> None:
    from dotenv import find_dotenv, load_dotenv

    from .llm import Cache, LLMClient
    from .runner import run

    load_dotenv(find_dotenv(usecwd=True))
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)

    config = RunConfig(
        models=args.models,
        variants=args.variants,
        n=args.n,
        seed=args.seed,
        exclude_ties=args.exclude_ties,
        calib_questions=args.calib_questions,
        concurrency=args.concurrency,
        out_dir=args.out,
    )
    cache = Cache(args.cache)
    client = LLMClient(cache, temperature=args.temperature, max_retries=args.max_retries, rpm=args.rpm)
    try:
        results = run(config, client)
    finally:
        cache.close()
    print((Path(args.out) / "report.md").read_text())
    if any(m["n_errors"] for m in results["metrics"]):
        sys.exit(2)


def _cmd_report(args: argparse.Namespace) -> None:
    from .report import render

    results = [json.loads(Path(p).read_text()) for p in args.results]
    md = render(results)
    out = Path(args.out) if args.out else Path(args.results[0]).with_name("report.md")
    out.write_text(md)
    print(md)
    print(f"wrote {out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="judgecal", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run judge variants on the eval split and write results")
    r.add_argument("--models", nargs="+", default=[DEFAULT_MODEL],
                   help=f"litellm model names, e.g. gemini/gemini-3.8-flash groq/llama-3.3-70b-versatile (default: {DEFAULT_MODEL})")
    r.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    r.add_argument("--n", type=int, default=200, help="eval pairs to sample (default: 200)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--exclude-ties", action="store_true", help="drop pairs whose human majority is a tie")
    r.add_argument("--calib-questions", type=int, default=8, help="questions held out for few-shot anchors (default: 8)")
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--rpm", type=float, default=None, help="max requests per minute across all threads")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-retries", type=int, default=6)
    r.add_argument("--cache", default=".judgecal_cache.sqlite")
    r.add_argument("--out", default="results", help="output directory (default: results)")
    r.set_defaults(func=_cmd_run)

    rep = sub.add_parser("report", help="render markdown from one or more results.json files")
    rep.add_argument("results", nargs="*", default=["results/results.json"],
                     help="results.json files; pass several to compare runs side by side")
    rep.add_argument("--out", help="markdown output path (default: report.md next to the first file)")
    rep.set_defaults(func=_cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
