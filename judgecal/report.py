"""Render results as a markdown report. Metrics are recomputed from saved records."""

from __future__ import annotations

from . import metrics
from .prompts import VARIANTS

SIDE_BY_SIDE = [
    ("accuracy", "Accuracy"),
    ("kappa", "Cohen's κ"),
    ("accuracy_swap_consistent", "Accuracy (swap-consistent)"),
    ("position_flip_rate", "Position flip rate ↓"),
    ("length_bias", "Length bias ↓"),
    ("unparseable_rate", "Unparseable ↓"),
]


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None) -> str:
    return "–" if x is None else f"{x:.3f}"


def _delta(x: float | None, base: float | None, pct: bool) -> str:
    if x is None or base is None:
        return "–"
    d = x - base
    return f"{100 * d:+.1f}pp" if pct else f"{d:+.3f}"


def _fmt(key: str, x: float | None) -> str:
    return _num(x) if key.startswith("kappa") else _pct(x)


def collect_rows(results_list: list[dict]) -> list[dict]:
    """Metric rows from one or more results files, sorted by model then variant order."""
    rows, seen_models = [], []
    for i, res in enumerate(results_list):
        for row in metrics.compute_all(res["records"], res["pairs"]):
            model = row["model"]
            if any(row["model"] == r["model"] and row["variant"] == r["variant"] for r in rows):
                model = f"{model} (file {i + 1})"
            row = {**row, "model": model}
            if model not in seen_models:
                seen_models.append(model)
            rows.append(row)
    vorder = {v: i for i, v in enumerate(VARIANTS)}
    rows.sort(key=lambda r: (seen_models.index(r["model"]), vorder.get(r["variant"], 99)))
    return rows


def render(results_list: list[dict]) -> str:
    rows = collect_rows(results_list)
    models = list(dict.fromkeys(r["model"] for r in rows))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    first = results_list[0]
    cfg, ds = first["config"], first["data"]
    out = ["# judgecal report", ""]

    out += [
        "## Setup",
        "",
        f"- Dataset: `lmsys/mt_bench_human_judgments` (human split, turn 1), {ds['unique_pairs']} unique pairs, "
        f"{ds['pairs_no_majority_dropped']} dropped for having no majority verdict",
        f"- Split by question: {ds['calib_questions']} calibration questions ({ds['calib_pairs']} pairs), "
        f"{ds['eval_questions']} eval questions",
        f"- Eval sample: {ds['eval_pairs_sampled']} pairs (seed {cfg['seed']}, n={cfg['n']}), "
        f"human labels {ds['eval_label_counts']}",
        f"- Ties: {'excluded' if cfg['exclude_ties'] else 'kept as a separate class'}",
        f"- Few-shot anchors (calibration split only): {', '.join(a['key'] + ' → ' + a['human'] for a in first['anchors']) or 'none'}",
        f"- Judge models: {', '.join(f'`{m}`' for m in models)}",
        "",
    ]
    if len(results_list) > 1:
        keysets = [set(r["pairs"]) for r in results_list]
        if any(k != keysets[0] for k in keysets[1:]):
            out += ["> **Warning:** these results files use different eval pairs, so they are not directly comparable.", ""]

    out += [
        "## Results",
        "",
        "| Model | Variant | Accuracy | Δ vs baseline | κ | Δ κ | Acc (swap-consistent) | Flip rate | Length bias | Unparseable | Errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    baseline = {r["model"]: r for r in rows if r["variant"] == "baseline"}
    for r in rows:
        b = baseline.get(r["model"]) if r["variant"] != "baseline" else None
        out.append(
            f"| `{r['model']}` | {r['variant']}{' ⚠ partial' if r['n_errors'] else ''} | {_pct(r['accuracy'])} | {_delta(r['accuracy'], b and b['accuracy'], True)} "
            f"| {_num(r['kappa'])} | {_delta(r['kappa'], b and b['kappa'], False)} | {_pct(r['accuracy_swap_consistent'])} "
            f"| {_pct(r['position_flip_rate'])} (n={r['position_flip_n']}) | {_pct(r['length_bias'])} (n={r['length_bias_n']}) "
            f"| {r['n_unparseable']}/{r['n_scored']} | {r['n_errors']} |"
        )
    out.append("")

    if len(models) > 1:
        by = {(r["model"], r["variant"]): r for r in rows}
        out += ["## Side by side", "", "| Metric | Variant | " + " | ".join(f"`{m}`" for m in models) + " |",
                "|---|---|" + "---|" * len(models)]
        for key, label in SIDE_BY_SIDE:
            for v in variants:
                cells = [_fmt(key, by[(m, v)][key]) if (m, v) in by else "–" for m in models]
                out.append(f"| {label} | {v} | " + " | ".join(cells) + " |")
        out.append("")

    errored = [r for r in rows if r["n_errors"]]
    if errored:
        out += [
            "> **Incomplete:** some calls failed after retries (see Errors). They are excluded from the metrics. "
            "Rerun the same command to fill them in; finished calls are served from the cache.",
            "",
        ]

    out += [
        "## Metric definitions",
        "",
        "- **Accuracy / κ**: judge verdict vs human majority over both answer orderings. Unparseable responses count as wrong "
        "(an extra `invalid` class for κ). API errors are excluded and reported separately.",
        "- **Acc (swap-consistent)**: one verdict per pair; if the two orderings disagree it becomes a tie (MT-Bench convention).",
        "- **Flip rate**: share of pairs whose verdict changes when A and B are swapped (both orderings parsed).",
        "- **Length bias**: among calls where the human majority did *not* pick the longer answer, how often the judge did.",
        "- **Δ**: change vs the `baseline` variant for the same model (pp = percentage points).",
        "",
    ]
    return "\n".join(out)
