## DS440 Project — LLM Stock Movement Explanations (from market data only)

This repo does two things:

- **Build a stock dataset** from Yahoo Finance OHLCV data (Open/High/Low/Close/Volume), then compute indicators.
- **Ask an LLM to explain / predict short-term movement** using *only those numeric indicators*, and save results in machine-readable files.

The key idea for the project: **we are not letting the LLM use news or outside context**—only the numbers we give it.

## What files matter (mental map)

- **`stock_data.py`**: downloads raw daily prices for a small list of tickers and writes:
  - `stock_raw_data.csv`
  - `stock_with_technical_indicators.csv`
  - `stock_with_technical_bayesian.csv`
- **`run_llm_stock.py`**: reads one of the CSVs, calls the LLM, and writes:
  - a **JSONL** file (one result per line), plus
  - a **flattened CSV** with the most useful fields pulled out for analysis.
- **`llm/`**: helper code
  - `llm/prompts.py`: defines the “prompt variants” and the exact JSON schema we demand back
  - `llm/providers.py`: provider wrappers + API key rotation (Groq / Gemini / Z.ai)
  - `llm/json_parsing.py`: “best effort” JSON extraction (we look for the last JSON object in the response)
  - `llm/call_once_stock.py`: one call + retry logic when parsing fails

## Pipeline overview (plain English)

### Step A — Build the dataset (`stock_data.py`)

1. Download daily price data for each ticker from Yahoo Finance.
2. Add **technical indicators** per ticker (examples: SMA, EMA, RSI, MACD, returns, rolling volatility).
3. Add simple **Bayesian-style indicators** (probability-like signals derived from recent up/down days and whether today is a “large move”).
4. Write out CSV files you can feed into the LLM step.

### Step B — Call the LLM (`run_llm_stock.py`)

1. Read the CSV into rows (each row is one trading day for one ticker).
2. Build a prompt from each row (or from a multi-day window of rows).
3. Ask the provider (Groq/Gemini/Z.ai) for a response.
4. Parse the response into **a JSON object** (or record a parse error).
5. Save:
   - **JSONL**: keeps the raw response + parsed JSON + errors for debugging
   - **CSV**: a flattened version that is easier to analyze in Excel/pandas

## Setup (Windows / PowerShell)

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set API keys (pick one provider)

You can either set environment variables *for your current terminal session*:

```powershell
$env:GROQ_API_KEY="..."
$env:GEMINI_API_KEY="..."
$env:ZAI_API_KEY="..."
```

Or create a local **`.env`** file in the repo root (recommended). Example `.env`:

```text
# Choose one or more keys. If you provide multiple keys, the code can rotate them.
GROQ_API_KEY=...
# GROQ_API_KEYS=key1,key2,key3

GEMINI_API_KEY=...
# GEMINI_API_KEYS=key1,key2

ZAI_API_KEY=...
# ZAI_API_KEYS=key1,key2

# Optional Z.ai settings
# ZAI_MODEL=glm-4.7-flash
# ZAI_THINKING=disabled
```

Notes:

- **Multiple keys / rotation**: set `GROQ_API_KEYS`, `GEMINI_API_KEYS`, or `ZAI_API_KEYS` as **comma- or newline-separated** lists. If those are present, the runner will rotate keys on common quota/auth errors.
- **No secrets in git**: `.env` should stay local (don’t commit it).

## How to run (copy/paste)

### 1) Build the CSV dataset

This produces the input files for the LLM step.

```bash
python stock_data.py
```

Expected outputs in the repo root:

- `stock_raw_data.csv`
- `stock_with_technical_indicators.csv`
- `stock_with_technical_bayesian.csv`

### 2) Run the LLM per day (one call per row)

Variant 1/2/3 correspond to different prompt formats (see `llm/prompts.py`). Variants 2 and 3 expect Bayesian columns, so they usually use `stock_with_technical_bayesian.csv`.

Example: **Gemini**, variant 3, only MSFT, first 25 rows:

```bash
python run_llm_stock.py --provider gemini --variant 3 --input stock_with_technical_bayesian.csv --where-ticker MSFT --limit 25
```

Example: **Groq**, variant 2, all tickers, first 50 rows:

```bash
python run_llm_stock.py --provider groq --variant 2 --input stock_with_technical_bayesian.csv --limit 50
```

Example: run **all variants (1,2,3)** back-to-back:

```bash
python run_llm_stock.py --provider gemini --all-variants --input stock_with_technical_bayesian.csv --where-ticker AAPL --limit 20
```

### 3) Run the LLM in “window mode” (one call per multi-day chunk)

Window mode groups consecutive days into a single prompt so the LLM reasons over a short history.

Example: 14 calendar-day windows (non-overlapping), MSFT:

```bash
python run_llm_stock.py --provider gemini --variant 3 --mode window --window-by calendar --window-calendar-days 14 --where-ticker MSFT --input stock_with_technical_bayesian.csv
```

Example: 10 trading-day windows (sliding by 5 trading days):

```bash
python run_llm_stock.py --provider groq --variant 2 --mode window --window-by trading --window-trading-days 10 --window-step-days 5 --where-ticker TSLA --input stock_with_technical_bayesian.csv
```

### 4) Understanding the outputs

For each run, the script writes two files:

- **JSONL**: `llm_outputs_<provider>_v<variant>_<timestamp>.jsonl`
  - each line is a JSON object containing:
    - `parsed`: the parsed JSON response (or `null` if parsing failed)
    - `parse_error`: why parsing failed (or `null`)
    - `raw_text`: the raw model output (useful to debug bad formatting)
- **CSV**: same name but `.csv`
  - includes helpful columns like `stance`, `direction`, `confidence`, `reason` when present

## Common issues / quick fixes

- **“Missing GROQ_API_KEY / GEMINI_API_KEY / ZAI_API_KEY”**: you didn’t set keys in your terminal or `.env`.
- **Lots of `parse_error`**: the model is not obeying “JSON-only”. The caller retries automatically, but some models/settings may still fail. Check `raw_text` in the JSONL to see what it returned.
- **Early rows have null indicators**: moving averages/volatility need a warm-up window. Use `--skip-missing` to avoid those rows:

```bash
python run_llm_stock.py --provider gemini --variant 3 --skip-missing --limit 50
```

## Notes / extra

- `make_prompt.py` is an older/simple prompt generator and isn’t required for the main pipeline (`run_llm_stock.py` is the main runner).

