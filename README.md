<div align="center">

# Chinese Fortune Telling Bench

**A benchmark for evaluating large language models on Chinese traditional fortune telling — Bazi (八字) and Ziwei Doushu (紫微斗数).**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Questions](https://img.shields.io/badge/Questions-160-green.svg)](./data/data.json)
[![Years](https://img.shields.io/badge/Years-2022--2025-orange.svg)](./data)

</div>

---

## Overview

Questions are multiple-choice and scored by exact match against a ground-truth answer. The corpus is sourced from the annual **Global Fortune Teller Competition** ([全球算命师大赛](https://hkjfma.org)) for 2022–2025. Raw sheets live under [data/raw/](./data/raw/).

| File | Description |
|---|---|
| [data/data.json](./data/data.json) | 160 normalized multiple-choice questions across twelve life aspects (career, health, marriage, children, wealth, …). |
| [data/fortune_api_results.json](./data/fortune_api_results.json) | Pre-computed Bazi and Ziwei charts (via [iztro](https://github.com/SylarLong/iztro)), keyed by `case_id` and joined to `data.json` at runtime when `--astro` is set. Isolates pure reasoning from chart derivation. |

> Filter by year with `--year`. Inspect the dataset with `--stats`.

---

## Installation

```bash
git clone https://github.com/DestinyLinker/MingLi-Bench.git
cd MingLi-Bench
pip install -r requirements.txt
```

---

## Configuration

The CLI reads API keys and defaults from a `.env` file. Copy the template and fill in only the providers you plan to use — empty or placeholder values are skipped automatically.

```bash
cp .env.example .env
```

<details>
<summary><b>Supported providers</b></summary>

```dotenv
# OpenRouter — one key, most models
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Native providers (only if you call them directly)
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://api.openai.com/v1   # override for OpenAI-compatible gateways
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com

# Doubao / Volcengine — both key and endpoint id are required
DOUBAO_API_KEY=...
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
DOUBAO_ENDPOINT_ID=ep-...

# Defaults (CLI flags can override)
TIMEOUT=60
MAX_WORKERS=5
MAX_TOKENS=8192
TEMPERATURE=0.0
```

</details>

`.env` is git-ignored. To run against a different key set, pass `--env-file /path/to/other.env`.

**Sanity check** once keys are in place:

```bash
python -m mingli_bench.cli --list-models   # supported model names
python -m mingli_bench.cli --stats         # dataset statistics
```

---

## Quick Start

> **Recommended defaults:** always pass `--cot` and `--astro`. Chain-of-Thought gives the model room to reason through the chart, and `--astro` injects pre-computed Bazi/Ziwei charts so scores reflect reasoning ability rather than date-to-chart conversion accuracy. Drop them only when you specifically want to ablate either effect.

There are two common routing modes.

### 1. Via OpenRouter (one key, most models)

Leave `--platform` unset and write the model name in OpenRouter's `provider/model` form. The CLI auto-detects the `/` and routes through OpenRouter.

```bash
python -m mingli_bench.cli --model openai/gpt-4o               --year 2025 --cot --astro --max-workers 8
python -m mingli_bench.cli --model anthropic/claude-sonnet-4-6 --year 2025 --cot --astro
python -m mingli_bench.cli --model google/gemini-2.5-pro       --year 2025 --cot --astro
python -m mingli_bench.cli --model deepseek/deepseek-r1        --year 2025 --cot --astro
```

### 2. Native provider

When the model name doesn't match auto-detection rules (e.g. a versioned Doubao endpoint id), or when you want to force an OpenAI-compatible gateway, pass `--platform` explicitly.

```bash
# Native Doubao / Volcengine
python -m mingli_bench.cli \
    --platform doubao --model doubao-seed-2-0-pro-260215 \
    --year 2025 --cot --astro --max-workers 8

# Any model name, routed through an OpenAI-compatible gateway (set via OPENAI_BASE_URL)
python -m mingli_bench.cli \
    --platform openai --model doubao-seed-2-0-pro-260215 \
    --year 2025 --cot --astro --max-workers 8
```

`--platform` accepts: `openai`, `openrouter`, `anthropic`, `google`, `deepseek`, `doubao`.

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--model, -m` | **required** | Model name. Contains `/` → treated as an OpenRouter id; otherwise provider is inferred from prefix (`gpt-*`, `claude-*`, `gemini-*`, `deepseek-*`, `doubao-*`). |
| `--platform` | inferred | Force routing platform, overriding prefix-based inference. |
| `--year, -y` | all | Evaluate one year only. Available: `2022`, `2023`, `2024`, `2025`. |
| `--max-workers` | `5` | Concurrent API calls. Raise to 8–16 if rate limits allow; lower on throttling. |
| `--cot` | off | Prepend a Chain-of-Thought instruction to the prompt. |
| `--astro` | off | Inject pre-computed Bazi/Ziwei charts from `data/fortune_api_results.json` into the prompt (model does not have to derive them from the birth date). |
| `--sample, -s N` | all | Evaluate only the first N questions. Useful as a smoke test. |
| `--categories, -c` | all | Filter by category, e.g. `--categories 事业 婚姻`. Available: 事业、健康、外貌、婚姻、子女、学业、官非、家庭、性格、灾劫、财运、运势. |
| `--shuffle-options` | off | Randomize option order per question to guard against position bias. |
| `--output-dir, -o` | `logs` | Directory for result files. |
| `--no-save` | off | Print to terminal only; do not write files. |
| `--env-file` | `.env` | Use a different env file. |
| `--list-models` | — | Print supported model names and exit. |
| `--stats` | — | Print dataset statistics (optionally filtered by `--year`) and exit. |

Full help: `python -m mingli_bench.cli --help`.

---

## Output

Each run writes three artifacts under `--output-dir` (default `logs/`):

| Artifact | Description |
|---|---|
| `<model>_results.json` | Per-question predictions, scoring, and aggregates. |
| `<model>_summary.txt`  | Headline numbers. |
| `<model>_responses/`   | Raw model responses, one file per question. |

---

## License

Released under the [MIT License](./LICENSE).

## Contact

- Issues: [github.com/DestinyLinker/MingLi-Bench/issues](https://github.com/DestinyLinker/MingLi-Bench/issues)
- Email: [help@destinylinker.com](mailto:help@destinylinker.com)
