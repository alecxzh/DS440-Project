from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ParsedJSON:
    obj: Optional[dict]
    error: Optional[str] = None


_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    text = _CODE_FENCE_RE.sub("", text or "")
    return text.replace("```", "")


def _extract_last_json_object(text: str) -> Optional[str]:
    """
    Best-effort: find the *last* JSON object in the text.
    This is robust to prefacing prose, and matches the "JSON on last line" instruction.
    """
    if not text:
        return None

    s = text.strip()

    # Fast path: last non-empty line is a JSON object
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if last.startswith("{") and last.endswith("}"):
            return last

    # General path: scan from end to find a balanced {...} object.
    depth = 0
    end_idx = None
    for i in range(len(s) - 1, -1, -1):
        ch = s[i]
        if ch == "}":
            if depth == 0:
                end_idx = i
            depth += 1
        elif ch == "{":
            if depth > 0:
                depth -= 1
                if depth == 0 and end_idx is not None:
                    return s[i : end_idx + 1].strip()
    return None


def parse_json_from_llm(text: str) -> ParsedJSON:
    cleaned = _strip_code_fences(text)
    blob = _extract_last_json_object(cleaned)
    if not blob:
        return ParsedJSON(obj=None, error="No JSON object found in response.")

    try:
        parsed = json.loads(blob)
    except Exception as e:
        return ParsedJSON(obj=None, error=f"JSON parse error: {e}")

    if not isinstance(parsed, dict):
        return ParsedJSON(obj=None, error="Parsed JSON is not an object.")

    return ParsedJSON(obj=parsed, error=None)


def coerce_direction(obj: dict) -> Optional[str]:
    d = obj.get("direction")
    if isinstance(d, str):
        d2 = d.strip().upper()
        # Variant 3 in this project uses a forced binary label.
        if d2 in {"UP", "DOWN"}:
            return d2
    return None


def coerce_confidence(obj: dict) -> Optional[float]:
    c = obj.get("confidence")
    if isinstance(c, (int, float)):
        c2 = float(c)
        if 0.0 <= c2 <= 1.0:
            return c2
    return None

