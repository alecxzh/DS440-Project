from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from llm.json_parsing import ParsedJSON, parse_json_from_llm
from llm.prompts import build_prompt, build_window_prompt, Variant
from llm.providers import BaseLLM


@dataclass(frozen=True)
class StockCallResult:
    prompt: str
    raw_text: str
    parsed: Optional[dict]
    parse_error: Optional[str]


_OVERLOAD_RE = re.compile(r"(429|rate limit|ratelimit|overload|temporarily overloaded|quota|throttl)", re.IGNORECASE)


def _should_backoff_llm_error(msg: str) -> bool:
    return bool(_OVERLOAD_RE.search(msg or ""))


def _retry_sleep_seconds(attempt: int) -> float:
    # attempt is 0-based; backoff grows modestly to avoid hammering overloaded APIs.
    base = min(30.0, 0.75 * (2**attempt))
    jitter = random.uniform(0.0, 0.35)
    return float(base + jitter)


def call_stock_once(
    llm: BaseLLM,
    row: Mapping[str, Any],
    variant: Variant,
    max_retries: int = 3,
) -> StockCallResult:
    """
    Builds the prompt for the given row+variant, calls the LLM, and parses JSON.

    Retries when:
    - provider raises an exception
    - JSON parsing fails
    """
    base_prompt = build_prompt(row=row, variant=variant)
    last_raw = ""
    last_err: Optional[str] = None

    for attempt in range(max_retries):
        prompt = base_prompt
        if attempt > 0:
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Your previous output did not follow the JSON-only rule. "
                + "Return ONLY a single JSON object on the last line, with no surrounding text.\n"
            )

        try:
            resp = llm.generate(prompt)
            last_raw = resp.text
        except Exception as e:
            last_err = f"LLM call error: {e}"
            if attempt < max_retries - 1 and _should_backoff_llm_error(last_err):
                time.sleep(_retry_sleep_seconds(attempt))
            continue

        parsed = parse_json_from_llm(last_raw)
        if parsed.obj is not None:
            return StockCallResult(prompt=prompt, raw_text=last_raw, parsed=parsed.obj, parse_error=None)

        last_err = parsed.error or "Unknown parse error"
        if attempt < max_retries - 1:
            time.sleep(_retry_sleep_seconds(attempt))

    return StockCallResult(prompt=base_prompt, raw_text=last_raw, parsed=None, parse_error=last_err)


def call_window_once(
    llm: BaseLLM,
    rows: Sequence[Mapping[str, Any]],
    variant: Variant,
    max_retries: int = 3,
) -> StockCallResult:
    """
    Same as call_stock_once but uses build_window_prompt (multi-day table in one prompt).
    """
    base_prompt = build_window_prompt(rows=rows, variant=variant)
    last_raw = ""
    last_err: Optional[str] = None

    for attempt in range(max_retries):
        prompt = base_prompt
        if attempt > 0:
            prompt = (
                base_prompt
                + "\n\nIMPORTANT: Your previous output did not follow the JSON-only rule. "
                + "Return ONLY a single JSON object on the last line, with no surrounding text.\n"
            )

        try:
            resp = llm.generate(prompt)
            last_raw = resp.text
        except Exception as e:
            last_err = f"LLM call error: {e}"
            if attempt < max_retries - 1 and _should_backoff_llm_error(last_err):
                time.sleep(_retry_sleep_seconds(attempt))
            continue

        parsed = parse_json_from_llm(last_raw)
        if parsed.obj is not None:
            return StockCallResult(prompt=prompt, raw_text=last_raw, parsed=parsed.obj, parse_error=None)

        last_err = parsed.error or "Unknown parse error"
        if attempt < max_retries - 1:
            time.sleep(_retry_sleep_seconds(attempt))

    return StockCallResult(prompt=base_prompt, raw_text=last_raw, parsed=None, parse_error=last_err)

