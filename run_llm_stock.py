from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd

from llm.call_once_stock import call_stock_once, call_window_once
from llm.providers import ProviderName, build_llm
from llm.prompts import Variant


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _default_output_path(provider: str, variant: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"llm_outputs_{provider}_v{variant}_{ts}.jsonl"

def _default_output_csv_path(jsonl_path: str) -> str:
    if jsonl_path.lower().endswith(".jsonl"):
        return jsonl_path[:-6] + ".csv"
    return jsonl_path + ".csv"


def _flatten_for_csv(
    provider: str,
    variant: int,
    idx: int,
    row: dict,
    parsed: dict | None,
    parse_error: str | None,
    raw_text: str,
    *,
    window_start: str | None = None,
    window_end: str | None = None,
    window_n: int | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "provider": provider,
        "variant": variant,
        "row_index": idx,
        "ticker": row.get("Ticker"),
        "date": row.get("Date"),
        "window_start": window_start or "",
        "window_end": window_end or "",
        "window_n_days": window_n if window_n is not None else "",
        "parse_error": parse_error,
        "raw_text": raw_text,
        "parsed_json": json.dumps(parsed, ensure_ascii=False) if parsed is not None else "",
        # common “analysis-friendly” fields (may be blank depending on variant)
        "stance": "",
        "direction": "",
        "confidence": "",
        "reason": "",
    }

    if isinstance(parsed, dict):
        if "stance" in parsed:
            out["stance"] = parsed.get("stance", "")
        if "direction" in parsed:
            out["direction"] = parsed.get("direction", "")
        if "confidence" in parsed:
            out["confidence"] = parsed.get("confidence", "")
        if "reason" in parsed:
            out["reason"] = parsed.get("reason", "")
        if "combined_narrative" in parsed and not out["reason"]:
            out["reason"] = parsed.get("combined_narrative", "")
    return out


def _row_ts(r: dict) -> pd.Timestamp:
    d = r.get("Date")
    if isinstance(d, pd.Timestamp):
        t = d
    else:
        t = pd.Timestamp(d)
    if pd.isna(t):
        return pd.NaT
    return t.normalize()


def _chunk_rows(rows: list[dict], window_days: int, step_days: int, min_window_days: int) -> list[list[dict]]:
    """Non-overlapping (or custom step) consecutive chunks by trading-row count."""
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    if step_days < 1:
        raise ValueError("step_days must be >= 1")
    chunks: list[list[dict]] = []
    i = 0
    while i < len(rows):
        chunk = rows[i : i + window_days]
        if len(chunk) >= min_window_days:
            chunks.append(chunk)
        i += step_days
    return chunks


def _chunk_rows_calendar(
    rows: list[dict],
    span_calendar_days: int,
    step_calendar_days: int,
    min_trading_rows: int,
) -> list[list[dict]]:
    """
    Chunks by **calendar time**: each window is [cursor, cursor + span_calendar_days)
    (half-open interval). Includes every trading row whose `Date` falls in that range.
    """
    if span_calendar_days < 1:
        raise ValueError("span_calendar_days must be >= 1")
    step = step_calendar_days if step_calendar_days > 0 else span_calendar_days

    valid = [r for r in rows if not pd.isna(_row_ts(r))]
    if not valid:
        return []
    valid.sort(key=_row_ts)

    chunks: list[list[dict]] = []
    cursor = _row_ts(valid[0])
    max_d = _row_ts(valid[-1])

    while cursor <= max_d:
        end = cursor + pd.Timedelta(days=span_calendar_days)
        chunk = [r for r in valid if cursor <= _row_ts(r) < end]
        if len(chunk) >= min_trading_rows:
            chunks.append(chunk)
        cursor = cursor + pd.Timedelta(days=step)
        # stop if cursor has moved past all data (no infinite loop)
        if cursor > max_d + pd.Timedelta(days=span_calendar_days):
            break

    return chunks


def main() -> None:
    # Load .env if present (API keys, etc.). Safe no-op if file missing.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Run Groq/Gemini over stock rows with structured JSON outputs.")
    ap.add_argument("--provider", choices=["groq", "gemini", "zai"], required=True)
    ap.add_argument("--variant", type=int, choices=[1, 2, 3], help="Which prompt variant to run (1/2/3).")
    ap.add_argument(
        "--all-variants",
        action="store_true",
        help="Run variants 1, 2, and 3 sequentially (writes separate outputs per variant).",
    )
    ap.add_argument(
        "--input",
        default="stock_with_technical_bayesian.csv",
        help="Input CSV (use stock_with_technical_indicators.csv for variant 1 if desired).",
    )
    ap.add_argument("--output", default=None, help="Output JSONL path (default: timestamped).")
    ap.add_argument("--output-csv", default=None, help="Optional flattened CSV output path.")
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows with missing required indicator fields (avoids early-window nulls).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional row limit (0 = all).")
    ap.add_argument("--where-ticker", default=None, help="Optional filter, e.g. MSFT.")
    ap.add_argument(
        "--mode",
        choices=["per-row", "window"],
        default="per-row",
        help="per-row: one LLM call per day. window: one call per multi-day window (see --window-by).",
    )
    ap.add_argument(
        "--window-by",
        choices=["calendar", "trading"],
        default="calendar",
        help="calendar: chunk by calendar span (e.g. 14 days). trading: chunk by N consecutive rows.",
    )
    ap.add_argument(
        "--window-calendar-days",
        type=int,
        default=14,
        help="Calendar span per window (days). Rows with Date in [start, start+N days) are included. Default 14 = 2 weeks.",
    )
    ap.add_argument(
        "--window-calendar-step-days",
        type=int,
        default=0,
        help="Advance the window start by this many calendar days. 0 = same as --window-calendar-days (non-overlapping).",
    )
    ap.add_argument(
        "--window-trading-days",
        type=int,
        default=10,
        help="When --window-by trading: window size = this many consecutive trading rows.",
    )
    ap.add_argument(
        "--window-step-days",
        type=int,
        default=0,
        help="When --window-by trading: step between windows (rows). 0 = same as --window-trading-days.",
    )
    ap.add_argument(
        "--min-window-days",
        type=int,
        default=5,
        help="Skip a window if it has fewer than this many trading rows (after filtering).",
    )
    args = ap.parse_args()

    provider: ProviderName = args.provider
    if not args.all_variants and args.variant is None:
        ap.error("Either provide --variant {1,2,3} or use --all-variants.")
    variants_to_run = [1, 2, 3] if args.all_variants else [int(args.variant)]

    df = pd.read_csv(args.input)
    if args.where_ticker:
        df = df[df["Ticker"] == args.where_ticker]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values(["Ticker", "Date"])

    llm = build_llm(provider)

    csv_fieldnames = [
        "provider",
        "variant",
        "row_index",
        "ticker",
        "date",
        "window_start",
        "window_end",
        "window_n_days",
        "stance",
        "direction",
        "confidence",
        "reason",
        "parse_error",
        "parsed_json",
        "raw_text",
    ]

    base_rows = df.to_dict(orient="records")
    limit_n = int(args.limit) if args.limit and args.limit > 0 else 0
    step_trading = int(args.window_step_days) if args.window_step_days and args.window_step_days > 0 else int(args.window_trading_days)
    cal_span = int(args.window_calendar_days)
    cal_step = int(args.window_calendar_step_days) if args.window_calendar_step_days and args.window_calendar_step_days > 0 else cal_span

    for v in variants_to_run:
        variant: Variant = v  # type: ignore[assignment]

        rows = base_rows
        if args.skip_missing:
            required = [
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
            ]
            if int(variant) in (2, 3):
                required += ["Posterior_Trend", "Prior_Up", "Posterior_Up", "Is_Large_Move"]

            def _row_ok(r: dict) -> bool:
                for k in required:
                    v = r.get(k)
                    if v is None:
                        return False
                    if isinstance(v, float) and v != v:  # NaN
                        return False
                return True

            rows = [r for r in base_rows if _row_ok(r)]

        if args.mode == "window":
            chunks = []
            if args.where_ticker:
                chunk_src = [r for r in rows if r.get("Ticker") == args.where_ticker]
                if args.window_by == "calendar":
                    chunks = _chunk_rows_calendar(
                        chunk_src,
                        span_calendar_days=cal_span,
                        step_calendar_days=cal_step,
                        min_trading_rows=int(args.min_window_days),
                    )
                else:
                    chunks = _chunk_rows(
                        chunk_src,
                        window_days=int(args.window_trading_days),
                        step_days=step_trading,
                        min_window_days=int(args.min_window_days),
                    )
            else:
                by_ticker: dict[str, list[dict]] = {}
                for r in rows:
                    t = r.get("Ticker")
                    if t is None:
                        continue
                    by_ticker.setdefault(str(t), []).append(r)
                for _t, trows in sorted(by_ticker.items()):
                    if args.window_by == "calendar":
                        chunks.extend(
                            _chunk_rows_calendar(
                                trows,
                                span_calendar_days=cal_span,
                                step_calendar_days=cal_step,
                                min_trading_rows=int(args.min_window_days),
                            )
                        )
                    else:
                        chunks.extend(
                            _chunk_rows(
                                trows,
                                window_days=int(args.window_trading_days),
                                step_days=step_trading,
                                min_window_days=int(args.min_window_days),
                            )
                        )
            if limit_n:
                chunks = chunks[:limit_n]
            total = len(chunks)
            if args.window_by == "calendar":
                suf = f"_c{cal_span}"
            else:
                suf = f"_t{int(args.window_trading_days)}"
            out_path = args.output or _default_output_path(provider, int(variant)).replace(".jsonl", f"{suf}.jsonl")
        else:
            if limit_n:
                rows = rows[:limit_n]
            total = len(rows)
            out_path = args.output or _default_output_path(provider, int(variant))

        out_csv_path = args.output_csv or _default_output_csv_path(out_path)

        ok = 0
        with open(out_path, "w", encoding="utf-8") as f, open(out_csv_path, "w", encoding="utf-8", newline="") as cf:
            cw = csv.DictWriter(cf, fieldnames=csv_fieldnames)
            cw.writeheader()

            if args.mode == "window":
                for idx, chunk in enumerate(chunks):
                    res = call_window_once(llm=llm, rows=chunk, variant=variant, max_retries=3)
                    w_start = str(chunk[0].get("Date"))
                    w_end = str(chunk[-1].get("Date"))
                    rep_row = {
                        "Ticker": chunk[0].get("Ticker"),
                        "Date": w_end,
                    }

                    record: Dict[str, Any] = {
                        "provider": provider,
                        "variant": int(variant),
                        "row_index": int(idx),
                        "ticker": rep_row.get("Ticker"),
                        "date": rep_row.get("Date"),
                        "window_start": w_start,
                        "window_end": w_end,
                        "window_n_days": len(chunk),
                        "parsed": res.parsed,
                        "parse_error": res.parse_error,
                        "raw_text": res.raw_text,
                    }

                    if res.parsed is not None:
                        ok += 1

                    f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                    cw.writerow(
                        _flatten_for_csv(
                            provider,
                            int(variant),
                            int(idx),
                            rep_row,
                            res.parsed,
                            res.parse_error,
                            res.raw_text,
                            window_start=w_start,
                            window_end=w_end,
                            window_n=len(chunk),
                        )
                    )

                    if (idx + 1) % 10 == 0 or (idx + 1) == total:
                        print(f"v{variant} window: {idx+1}/{total} done | parsed_ok={ok}")
            else:
                for idx, row in enumerate(rows):
                    res = call_stock_once(llm=llm, row=row, variant=variant, max_retries=3)

                    record = {
                        "provider": provider,
                        "variant": int(variant),
                        "row_index": int(idx),
                        "ticker": row.get("Ticker"),
                        "date": row.get("Date"),
                        "window_start": "",
                        "window_end": "",
                        "window_n_days": "",
                        "parsed": res.parsed,
                        "parse_error": res.parse_error,
                        "raw_text": res.raw_text,
                    }

                    if res.parsed is not None:
                        ok += 1

                    f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
                    cw.writerow(_flatten_for_csv(provider, int(variant), int(idx), row, res.parsed, res.parse_error, res.raw_text))

                    if (idx + 1) % 25 == 0 or (idx + 1) == total:
                        print(f"v{variant}: {idx+1}/{total} done | parsed_ok={ok}")

        print(f"v{variant}: wrote {total} records to {out_path} (parsed_ok={ok}).")
        print(f"v{variant}: wrote flattened CSV to {out_csv_path}.")


if __name__ == "__main__":
    main()

