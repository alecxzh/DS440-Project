from __future__ import annotations

from typing import Literal, Mapping, Any, Sequence


Variant = Literal[1, 2, 3]

# Window mode: one prompt per chunk of consecutive trading days (default ~2 weeks ≈ 10 sessions).
WINDOW_TABLE_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "EMA_10",
    "EMA_20",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "Return",
    "Volatility_10d",
    "Posterior_Trend",
    "Prior_Up",
    "Posterior_Up",
    "Is_Large_Move",
]


def _fmt(v: Any) -> str:
    # Normalize missing values so the model never sees literal "nan".
    # Using JSON-ish null keeps downstream instructions consistent.
    if v is None:
        return "null"
    # NaN is the only value where v != v is True (works for float nan)
    if isinstance(v, float) and v != v:  # noqa: PLR0124
        return "null"
    return str(v)


def build_prompt(row: Mapping[str, Any], variant: Variant) -> str:
    """
    Builds prompts aligned to the PDF, but enforces JSON outputs for easier parsing.
    """
    ticker = _fmt(row.get("Ticker"))
    date = _fmt(row.get("Date"))

    # OHLCV
    open_ = _fmt(row.get("Open"))
    high = _fmt(row.get("High"))
    low = _fmt(row.get("Low"))
    close = _fmt(row.get("Close"))
    vol = _fmt(row.get("Volume"))

    # Technicals
    sma10 = _fmt(row.get("SMA_10"))
    sma20 = _fmt(row.get("SMA_20"))
    sma50 = _fmt(row.get("SMA_50"))
    ema10 = _fmt(row.get("EMA_10"))
    ema20 = _fmt(row.get("EMA_20"))
    rsi14 = _fmt(row.get("RSI_14"))
    macd = _fmt(row.get("MACD"))
    macd_signal = _fmt(row.get("MACD_signal"))
    macd_hist = _fmt(row.get("MACD_hist"))
    ret = _fmt(row.get("Return"))
    vol10 = _fmt(row.get("Volatility_10d"))

    # Bayesian
    post_trend = _fmt(row.get("Posterior_Trend"))
    prior_up = _fmt(row.get("Prior_Up"))
    post_up = _fmt(row.get("Posterior_Up"))
    is_large = _fmt(row.get("Is_Large_Move"))

    header = (
        f"Stock: {ticker}\n"
        f"Date: {date}\n"
        f"OHLC: open={open_}, high={high}, low={low}, close={close}\n"
        f"Volume: {vol}\n"
        f"Trend Indicators:\n"
        f"SMA_10={sma10}, SMA_20={sma20}, SMA_50={sma50}\n"
        f"EMA_10={ema10}, EMA_20={ema20}\n"
        f"MACD={macd}, MACD_signal={macd_signal}, MACD_hist={macd_hist}\n"
        f"Momentum & Risk:\n"
        f"RSI_14={rsi14}, Daily_Return={ret}, Volatility_10d={vol10}\n"
    )

    if variant == 1:
        return (
            "You are a careful financial data explainer.\n"
            "You ONLY use the numeric fields provided below.\n"
            "You do NOT use or mention news, earnings, macroeconomics, fundamentals, analyst ratings, or any external information.\n"
            "If the indicators are conflicting or weak, you explicitly say so and lower confidence.\n\n"
            "Missing values appear as null. Treat null as 'unavailable' and do NOT quote or mention null/nan in the output.\n\n"
            "INPUT (DATA)\n"
            + header
            + "\nDEFINITIONS (use these)\n"
            + '- "BULLISH": indicators collectively lean toward upward drift over the next few days.\n'
            + '- "BEARISH": indicators collectively lean toward downward drift over the next few days.\n'
            + '- "NEUTRAL": mixed/flat signals; no clear directional edge.\n\n'
            + "TASK\n"
            + "1. Describe the short-term trend and momentum in simple terms (no jargon unless briefly defined).\n"
            + '2. Choose ONE stance: "BULLISH", "BEARISH", or "NEUTRAL".\n'
            + "3. Give 2–3 plausible, data-grounded reasons the price could move up or down in the next few days.\n"
            + "   Each reason must reference specific fields from the INPUT (e.g., SMA relationships, MACD vs signal, RSI level, recent return/volatility).\n\n"
            + "RESPONSE FORMAT (MANDATORY)\n"
            + "- Output exactly ONE JSON object on the LAST line.\n"
            + "- Output NOTHING else (no markdown, no code fences).\n"
            + "- All strings must be plain text.\n\n"
            + "JSON SCHEMA (must match)\n"
            + '{\n'
            + '  "trend_summary": "<1-3 sentences>",\n'
            + '  "stance": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            + '  "confidence": <number between 0 and 1>,\n'
            + '  "reasons": ["<reason 1>", "<reason 2>", "<reason 3 optional>"],\n'
            + '  "grounding": ["<field=value>", "<field=value>", "..."]\n'
            + "}\n"
        )

    if variant == 2:
        return (
            "You are a careful financial data explainer.\n"
            "You ONLY use the numeric fields provided below.\n"
            "You do NOT use or mention news, earnings, macroeconomics, fundamentals, analyst ratings, or any external information.\n"
            "You must explicitly interpret the Bayesian indicators as probabilities and relate them to the technical indicators.\n"
            "If the indicators are conflicting or weak, you explicitly say so and lower confidence.\n\n"
            "Missing values appear as null. Treat null as 'unavailable' and do NOT quote or mention null/nan in the output.\n\n"
            "INPUT (DATA)\n"
            + header
            + "Bayesian Indicators:\n"
            + f"Posterior_Trend={post_trend}\n"
            + f"Prior_Up={prior_up}\n"
            + f"Posterior_Up={post_up}\n"
            + f"Is_Large_Move={is_large}\n"
            + "\nDEFINITIONS (use these)\n"
            + '- "Posterior_Trend": probability-like score summarizing recent up-days vs down-days (higher = more up-trend evidence).\n'
            + '- "Prior_Up": baseline/upside probability before accounting for today’s return shock.\n'
            + '- "Posterior_Up": updated upside probability after accounting for today’s move (when a large-move event occurs).\n'
            + '- "Is_Large_Move": whether today’s return magnitude is unusually large (event flag).\n'
            + '- "BULLISH": combined signals lean toward upward drift over the next few days.\n'
            + '- "BEARISH": combined signals lean toward downward drift over the next few days.\n'
            + '- "NEUTRAL": mixed/flat signals; no clear directional edge.\n\n'
            + "TASK\n"
            + "1. In plain language, explain what Posterior_Trend, Prior_Up, and Posterior_Up imply about upside vs downside.\n"
            + "2. If Is_Large_Move is true, explain how a large one-day move should change the outlook versus recent days.\n"
            + "   If Is_Large_Move is false, say that no special event update is implied.\n"
            + "3. Combine the technical + Bayesian indicators into one short narrative about likely behavior over the next few days.\n"
            + '4. Choose ONE stance: "BULLISH", "BEARISH", or "NEUTRAL".\n\n'
            + "RESPONSE FORMAT (MANDATORY)\n"
            + "- Output exactly ONE JSON object on the LAST line.\n"
            + "- Output NOTHING else (no markdown, no code fences).\n"
            + "- All strings must be plain text.\n\n"
            + "JSON SCHEMA (must match)\n"
            + "{\n"
            + '  "upside_downside_summary": "<1-3 sentences>",\n'
            + '  "large_move_effect": "<1-3 sentences>",\n'
            + '  "combined_narrative": "<2-5 sentences>",\n'
            + '  "stance": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            + '  "confidence": <number between 0 and 1>,\n'
            + '  "grounding": ["<field=value>", "<field=value>", "..."]\n'
            + "}\n"
        )

    if variant == 3:
        return (
            "You are a careful financial indicator-based forecaster.\n"
            "You ONLY use the numeric fields provided below.\n"
            "You do NOT use or mention news, earnings, macroeconomics, fundamentals, analyst ratings, or any external information.\n"
            "You must produce (1) a 3-day direction label and (2) a concise justification grounded in the inputs.\n"
            "You MUST choose a direction even when signals are mixed; in that case, choose the slightly more likely direction and lower confidence.\n\n"
            "Missing values appear as null. Treat null as 'unavailable' and do NOT quote or mention null/nan in the output.\n\n"
            "INPUT (DATA)\n"
            + header
            + "Bayesian Indicators:\n"
            + f"Posterior_Trend={post_trend}\n"
            + f"Prior_Up={prior_up}\n"
            + f"Posterior_Up={post_up}\n"
            + f"Is_Large_Move={is_large}\n"
            + "\nREASONING STRUCTURE (use these nodes internally)\n"
            + 'Node 1: "RecentTrend" — infer from SMA/EMA relationships, MACD vs signal, RSI_14, and recent Return/Volatility_10d.\n'
            + 'Node 2: "EventToday" — interpret Is_Large_Move and today’s Return sign/magnitude.\n'
            + 'Node 3: "BeliefUptrend" — interpret Prior_Up vs Posterior_Up and Posterior_Trend as probability-like evidence.\n'
            + 'Node 4: "PriceMoveToday" — connect today’s Return to momentum/reversion risk given Volatility_10d.\n\n'
            + "TASK\n"
            + '1. Choose exactly one direction for the next ~3 days: "UP" or "DOWN".\n'
            + "- UP: you believe rising is more likely than falling.\n"
            + "- DOWN: you believe falling is more likely than rising.\n"
            + "2. Provide a short, single-paragraph justification that references BOTH:\n"
            + "   - at least 2 technical fields (e.g., SMA/EMA/MACD/RSI/Return/Volatility_10d)\n"
            + "   - at least 1 Bayesian field (Posterior_Trend/Prior_Up/Posterior_Up/Is_Large_Move)\n\n"
            + "RESPONSE FORMAT (MANDATORY)\n"
            + "- Output exactly ONE JSON object on the LAST line.\n"
            + "- Output NOTHING else (no markdown, no code fences).\n\n"
            + "JSON SCHEMA (must match)\n"
            + "{\n"
            + '  "direction": "UP" | "DOWN",\n'
            + '  "confidence": <number between 0 and 1>,\n'
            + '  "reason": "<single short paragraph>",\n'
            + '  "grounding": ["<field=value>", "<field=value>", "..."]\n'
            + "}\n"
        )

    raise ValueError(f"Unknown variant: {variant}")


def _window_table_block(rows: Sequence[Mapping[str, Any]], include_bayesian: bool) -> str:
    cols = [c for c in WINDOW_TABLE_COLUMNS if include_bayesian or c not in (
        "Posterior_Trend", "Prior_Up", "Posterior_Up", "Is_Large_Move",
    )]
    header = "|".join(cols)
    lines = [header]
    for r in rows:
        lines.append("|".join(_fmt(r.get(c)) for c in cols))
    return "\n".join(lines)


def build_window_prompt(rows: Sequence[Mapping[str, Any]], variant: Variant) -> str:
    """
    One prompt containing all daily rows in a consecutive window for one ticker.
    The model must reason from the full window (not a single day).
    """
    if not rows:
        raise ValueError("Window requires at least one row.")
    ticker = _fmt(rows[0].get("Ticker"))
    for r in rows[1:]:
        if _fmt(r.get("Ticker")) != ticker:
            raise ValueError("All rows in a window must have the same Ticker.")

    start_d = _fmt(rows[0].get("Date"))
    end_d = _fmt(rows[-1].get("Date"))
    n = len(rows)
    include_bayesian = variant in (2, 3)

    table = _window_table_block(rows, include_bayesian=include_bayesian)
    window_header = (
        f"WINDOW (single ticker)\n"
        f"Ticker: {ticker}\n"
        f"Trading days in window: {n}\n"
        f"First date: {start_d}\n"
        f"Last date: {end_d}\n\n"
        "DAILY DATA (one row per trading day; columns are pipe-separated)\n"
        f"{table}\n"
    )

    if variant == 1:
        return (
            "You are a careful financial data explainer.\n"
            "You ONLY use the numeric fields in the WINDOW table below.\n"
            "You do NOT use or mention news, earnings, macroeconomics, fundamentals, analyst ratings, or any external information.\n"
            "Missing values appear as null. Treat null as 'unavailable' and do NOT quote or mention null/nan in the output.\n\n"
            + window_header
            + "\nTASK (WINDOW)\n"
            + "1. Summarize how price and momentum evolved over this entire window (not just the last day).\n"
            + '2. Choose ONE stance for the *next few trading days after the last date in the window*: "BULLISH", "BEARISH", or "NEUTRAL".\n'
            + "3. Give 2–3 reasons grounded in patterns across the window (e.g., moving-average crosses, MACD/RSI evolution, return/volatility).\n\n"
            + "RESPONSE FORMAT (MANDATORY)\n"
            + "- Output exactly ONE JSON object on the LAST line.\n"
            + "- Output NOTHING else (no markdown, no code fences).\n\n"
            + "JSON SCHEMA (must match)\n"
            + "{\n"
            + '  "window_summary": "<2-4 sentences>",\n'
            + '  "stance": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            + '  "confidence": <number between 0 and 1>,\n'
            + '  "reasons": ["<reason 1>", "<reason 2>", "<reason 3 optional>"],\n'
            + '  "grounding": ["<field=value>", "..."]\n'
            + "}\n"
        )

    if variant == 2:
        return (
            "You are a careful financial data explainer.\n"
            "You ONLY use the numeric fields in the WINDOW table below.\n"
            "You do NOT use or mention news, earnings, macroeconomics, fundamentals, analyst ratings, or any external information.\n"
            "You must relate Bayesian indicators to technical indicators across the window.\n"
            "Missing values appear as null. Treat null as 'unavailable' and do NOT quote or mention null/nan in the output.\n\n"
            + window_header
            + "\nTASK (WINDOW)\n"
            + "1. Explain how Posterior_Trend, Prior_Up, and Posterior_Up evolve over the window and what they imply for upside vs downside.\n"
            + "2. Note any large-move days (Is_Large_Move) and how they affect the narrative versus quieter days.\n"
            + "3. Combine technical + Bayesian into one narrative about likely behavior over the *next few trading days after the last date*.\n"
            + '4. Choose ONE stance: "BULLISH", "BEARISH", or "NEUTRAL".\n\n'
            + "RESPONSE FORMAT (MANDATORY)\n"
            + "- Output exactly ONE JSON object on the LAST line.\n"
            + "- Output NOTHING else (no markdown, no code fences).\n\n"
            + "JSON SCHEMA (must match)\n"
            + "{\n"
            + '  "upside_downside_summary": "<2-4 sentences>",\n'
            + '  "large_move_summary": "<1-3 sentences>",\n'
            + '  "combined_narrative": "<2-5 sentences>",\n'
            + '  "stance": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            + '  "confidence": <number between 0 and 1>,\n'
            + '  "grounding": ["<field=value>", "..."]\n'
            + "}\n"
        )

    if variant == 3:
        return (
            "You are a careful financial indicator-based forecaster.\n"
            "You ONLY use the numeric fields in the WINDOW table below.\n"
            "You do NOT use or mention news, earnings, macroeconomics, fundamentals, analyst ratings, or any external information.\n"
            "You MUST choose a binary direction even when signals are mixed; pick the slightly more likely direction and lower confidence.\n"
            "Missing values appear as null. Treat null as 'unavailable' and do NOT quote or mention null/nan in the output.\n\n"
            + window_header
            + "\nTASK (WINDOW)\n"
            + "1. Synthesize trend, momentum, volatility, and Bayesian evolution across the *entire window*.\n"
            + '2. Choose exactly one label for price direction over the *next ~5 trading days after the last date in the window*: "UP" or "DOWN".\n'
            + "3. Justify using both technical patterns across the window and Bayesian fields (reference at least two distinct dates or aggregates).\n\n"
            + "RESPONSE FORMAT (MANDATORY)\n"
            + "- Output exactly ONE JSON object on the LAST line.\n"
            + "- Output NOTHING else (no markdown, no code fences).\n\n"
            + "JSON SCHEMA (must match)\n"
            + "{\n"
            + '  "window_summary": "<2-4 sentences>",\n'
            + '  "direction": "UP" | "DOWN",\n'
            + '  "confidence": <number between 0 and 1>,\n'
            + '  "reason": "<single short paragraph>",\n'
            + '  "grounding": ["<field=value>", "..."]\n'
            + "}\n"
        )

    raise ValueError(f"Unknown variant: {variant}")

