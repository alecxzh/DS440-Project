## DS440 Project — LLM Stock Movement Explanations

This repo generates LLM explanations of short-term stock movement from **only** market data features (OHLCV + technical indicators, optionally Bayesian indicators). It supports **Groq** and **Gemini** and produces **structured JSON outputs** for easier evaluation.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Set API keys (PowerShell examples):

```powershell
$env:GROQ_API_KEY="..."
$env:GEMINI_API_KEY="..."
```

Or use a local `.env` file (recommended). Copy `env.example` to `.env` and fill in keys.
You can also set multiple keys for rotation:
- `GROQ_API_KEYS` (comma or newline separated)
- `GEMINI_API_KEYS` (comma or newline separated)

## Run the LLM (structured outputs)

Variant 1/2/3 correspond to the update PDF; this runner enforces **JSON-only outputs** so you can parse results automatically.

Example (Gemini, Variant 3, first 25 rows of MSFT):

```bash
python run_llm_stock.py --provider gemini --variant 3 --input stock_with_technical_bayesian.csv --where-ticker MSFT --limit 25
```

Example (Groq, Variant 2, first 50 rows across all tickers):

```bash
python run_llm_stock.py --provider groq --variant 2 --input stock_with_technical_bayesian.csv --limit 50
```

Outputs are written as **JSONL** (`llm_outputs_<provider>_v<variant>_<timestamp>.jsonl`) with fields:
- `parsed`: parsed JSON object (or null)
- `parse_error`: parse error string (or null)
- `raw_text`: raw LLM response (for debugging)

## Notes
- **No secrets** are stored in this repo. Use environment variables or a local `.env` (ignored by git).
- The dataset pipeline is in `stock_data.py` and produces `stock_with_technical_indicators.csv` and `stock_with_technical_bayesian.csv`.

