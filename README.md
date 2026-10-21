# judgecal

Measures how well an LLM-as-judge agrees with human judgments on MT-Bench, and how much calibration (a rubric, then rubric plus few-shot anchors) improves that agreement.

## Setup

```bash
git clone https://github.com/Tejashree0405/judgecal.git
cd judgecal
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

API keys are read from `.env` in the working directory. litellm picks up the standard names: `GEMINI_API_KEY`, `GROQ_API_KEY`, `OPENROUTER_API_KEY`. Ollama needs no key, just a running server.

## Usage

```bash
# Default: gemini/gemini-3.8-flash, 200 eval pairs, all three variants
judgecal run

# Two judges side by side in one report
judgecal run --models gemini/gemini-3.8-flash groq/llama-3.3-70b-versatile

# Other options
judgecal run --n 50 --seed 1 --exclude-ties --variants baseline rubric --rpm 10 --out results/no_ties

# Re-render a report, or merge separate runs into one comparison
judgecal report results/results.json
judgecal report results/gemini/results.json results/groq/results.json --out results/compare.md
```

Model names use litellm's `provider/model` format, for example `gemini/...`, `groq/...`, `openrouter/...`, or `ollama/llama3.1`.

Each `run` writes `results.json` (config, data stats, metrics, and every raw judge response) and `report.md` to `--out`.

## How it works

**Data.** The `lmsys/mt_bench_human_judgments` human split, turn 1 only. Many pairs appear in both orders, (m1, m2) and (m2, m1), so model names are put into sorted order (flipping the winner to match) before votes are pooled. Each pair gets a strict-plurality majority verdict. Pairs where the top two labels have equal counts, such as one A vote and one B vote, have no majority and are dropped; the report shows how many. Ties stay as their own class unless you pass `--exclude-ties`.

**Split.** The split is by `question_id`. `--calib-questions` (default 8) questions form the calibration split and the rest form the eval split, so no question appears in both. The eval set is a seeded sample of `--n` pairs.

**Judge variants**

| Variant | Prompt |
|---|---|
| `baseline` | "Which answer is better: A, B, or tie?" Verdict only. |
| `rubric` | Helpfulness, correctness, depth, instruction-following. Brief reasoning, then a verdict. |
| `rubric_fewshot` | The rubric plus 3 anchor examples (one each of A, B and tie) with human verdicts, taken only from the calibration split. |

**Metrics.** Every pair is judged twice, once with A and B swapped. Verdicts are mapped back to the original order.

- **Accuracy and Cohen's κ** compare the judge's verdict with the human majority, over both orderings.
- **Swap-consistent accuracy** follows the MT-Bench convention: if the two orderings disagree, the pair counts as a tie.
- **Position flip rate** is the share of pairs whose verdict changes when the answers are swapped.
- **Length bias** looks at calls where humans did not pick the longer answer, and reports how often the judge did.
- **Unparseable responses** are counted, reported, and scored as wrong. API failures that persist after retries are excluded from the metrics and listed in the report's Errors column.

## Cost and rate limits

- **Cache.** Every response is stored in `.judgecal_cache.sqlite`, keyed on model, messages and temperature, so reruns make no API calls. Verdicts are parsed when the report is built, so a parser change applies to old results without new calls.
- **Retries.** Rate-limit and server errors are retried with exponential backoff and jitter.
- **Daily quota.** If 5 calls in a row fail even after retries, the run stops early and saves partial results. Rerun the same command later to continue; finished calls come from the cache.
- **Call volume.** A full default run makes 200 pairs × 3 variants × 2 orderings = 1,200 calls per model. Use `--rpm` to stay under a per-minute limit.
- **Free tiers are small.** In October 2026, Gemini's free tier allowed only about 20 requests per day per model. Groq's limits are listed at about 200K tokens per day for its larger models. An 83-pair run (`--n 83`, 498 calls) uses about 560K tokens, and the few-shot variant uses about 3× the tokens of the others. On a free tier, either use a small `--n` or run the same command once a day until it finishes.
- **A daily quota error** stops the run right away instead of retrying.

## Tests

```bash
.venv/bin/pytest
```

The tests use a fake LLM and synthetic data, so they need no network access or API keys.
