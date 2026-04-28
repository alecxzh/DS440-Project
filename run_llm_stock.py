from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

import pandas as pd

from llm.call_once_stock import call_stock_once, call_window_once
from llm.providers import ProviderName, build_llm
from llm.prompts import Variant


_RL_RE = re.compile(r"(429|rate limit|ratelimit|overload|temporarily overloaded|quota|throttl)", re.IGNORECASE)

T = TypeVar("T")


def _with_wait_heartbeat(label: str, every_s: float, fn: Callable[[], T]) -> T:
    """
    LLM calls can block for a long time with no intermediate prints; this keeps the terminal informative.
    """
    stop = threading.Event()
    start = time.time()

    def _loop() -> None:
        while not stop.wait(timeout=float(every_s)):
            dt = int(time.time() - start)
            print(f"{label} | still waiting... {dt}s", flush=True)

    th = threading.Thread(target=_loop, name="llm-wait-heartbeat", daemon=True)
    th.start()
    try:
        return fn()
    finally:
        stop.set()
        th.join(timeout=1.0)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, pd.Timestamp):
        return None if pd.isna(obj) else obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _default_output_path(provider: str, variant: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"llm_outputs_{provider}_v{variant}_{ts}.jsonl"


def _safe_slug(s: str) -> str:
    """
    Keep filenames readable across OSes.
    - allow alnum, dash, underscore
    - collapse other chars to '-'
    """
    out = []
    prev_dash = False
    for ch in (s or "").strip():
        ok = ch.isalnum() or ch in {"-", "_"}
        if ok:
            out.append(ch)
            prev_dash = False
            continue
        if not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "NA"


def _default_output_path_v2(
    *,
    provider: str,
    variant: int,
    mode: str,
    input_path: str,
    ticker: str | None,
    window_by: str,
    cal_span: int,
    cal_step: int,
    trading_n: int,
    trading_step: int,
) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    inp = os.path.splitext(os.path.basename(input_path or ""))[0] or "input"
    inp = _safe_slug(inp)
    t = _safe_slug(ticker) if ticker else "ALL"

    if mode == "window":
        if window_by == "calendar":
            win = f"win-c{int(cal_span)}"
            if int(cal_step) != int(cal_span):
                win += f"-s{int(cal_step)}"
        else:
            win = f"win-t{int(trading_n)}"
            if int(trading_step) != int(trading_n):
                win += f"-s{int(trading_step)}"
    else:
        win = "per-row"

    return f"llm_{provider}_v{int(variant)}_{win}_{t}_{inp}_{ts}.jsonl"

def _default_output_csv_path(jsonl_path: str) -> str:
    if jsonl_path.lower().endswith(".jsonl"):
        return jsonl_path[:-6] + ".csv"
    return jsonl_path + ".csv"


def _window_key(ticker: str, window_start: Any, window_end: Any) -> Tuple[str, str, str, int]:
    t = str(ticker)
    ws = pd.to_datetime(window_start, errors="coerce")
    we = pd.to_datetime(window_end, errors="coerce")
    ws_s = "" if pd.isna(ws) else str(ws.normalize())
    we_s = "" if pd.isna(we) else str(we.normalize())
    return (t, ws_s, we_s, -1)


def _infer_resume_state_from_jsonl(
    jsonl_path: str,
    *,
    expected_provider: str,
    expected_variant: int,
    work_per_ticker: dict[str, list[Any]],
    mode: str,
) -> Tuple[int, Optional[str], int, set[Tuple[str, str, str, int]]]:
    """
    Returns:
    - line_count: number of JSONL lines for this provider+variant (continue row_index from here)
    - resume_ticker: first ticker (in sorted tickers order) that still has unfinished work
    - resume_local_completed: number of leading planned units already successfully completed for resume_ticker
    - seen_ok_keys: successful window keys seen in the file (used for prefix matching)
    """
    line_count = 0
    seen_ok: set[Tuple[str, str, str, int]] = set()

    with open(jsonl_path, "r", encoding="utf-8") as rf:
        for line in rf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if str(obj.get("provider")) != str(expected_provider):
                continue
            if int(obj.get("variant") or -1) != int(expected_variant):
                continue

            line_count += 1
            if obj.get("parse_error") is not None:
                continue
            if obj.get("parsed") is None:
                continue

            if mode == "window":
                k = _window_key(str(obj.get("ticker") or ""), obj.get("window_start"), obj.get("window_end"))
                n = obj.get("window_n_days")
                try:
                    nn = int(n) if n is not None and str(n).strip() != "" else -1
                except Exception:
                    nn = -1
                seen_ok.add((k[0], k[1], k[2], nn))

    tickers = sorted(work_per_ticker.keys())
    resume_ticker: Optional[str] = None
    resume_local_completed = 0

    if mode != "window":
        # Per-row resume needs a different keying scheme; keep safe default: do not auto-skip.
        return int(line_count), None, 0, seen_ok

    for t in tickers:
        units = work_per_ticker.get(t, [])
        prefix_done = 0
        for ch in units:
            if not ch:
                break
            ws = ch[0].get("Date")
            we = ch[-1].get("Date")
            key = _window_key(str(t), ws, we)
            key2 = (key[0], key[1], key[2], int(len(ch)))
            if key2 in seen_ok:
                prefix_done += 1
                continue
            break

        if prefix_done < len(units):
            resume_ticker = t
            resume_local_completed = int(prefix_done)
            break

    # If everything is already complete, still return the last ticker with full completion.
    if resume_ticker is None and tickers:
        resume_ticker = tickers[-1]
        resume_local_completed = len(work_per_ticker.get(resume_ticker, []))

    return int(line_count), resume_ticker, int(resume_local_completed), seen_ok


def _maybe_rate_limit_cooldown(parse_error: str | None) -> None:
    if not parse_error:
        return
    if not _RL_RE.search(parse_error):
        return
    # Extra cooldown between windows/rows when the provider is throttling us.
    time.sleep(min(60.0, 2.0 + random.uniform(0.0, 1.25)))


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
        "direction": "",
        "confidence": "",
        "reason": "",
    }

    if isinstance(parsed, dict):
        if "direction" in parsed:
            out["direction"] = parsed.get("direction", "")
        if "confidence" in parsed:
            out["confidence"] = parsed.get("confidence", "")
        # normalize the main narrative field for analysis
        if "reason" in parsed and parsed.get("reason"):
            out["reason"] = parsed.get("reason", "")
        elif "combined_narrative" in parsed and parsed.get("combined_narrative"):
            out["reason"] = parsed.get("combined_narrative", "")
        elif "trend_summary" in parsed or "reasons" in parsed:
            parts: list[str] = []
            ts = parsed.get("trend_summary")
            if isinstance(ts, str) and ts.strip():
                parts.append(ts.strip())
            rs = parsed.get("reasons")
            if isinstance(rs, list):
                reasons = [str(x).strip() for x in rs if str(x).strip()]
                if reasons:
                    parts.append("Reasons: " + " | ".join(reasons))
            out["reason"] = " ".join(parts).strip()
        elif "window_summary" in parsed and parsed.get("window_summary"):
            out["reason"] = parsed.get("window_summary", "")
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
    # On some Windows setups, stdout can be fully-block-buffered even in a terminal,
    # which makes progress prints appear "stuck" until a large buffer fills.
    for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        reconf = getattr(_stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(line_buffering=True)
            except Exception:
                pass

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
        "--resume-jsonl",
        default=None,
        help="Resume from an existing JSONL output (use with --resume). Skips already-finished tickers/chunks.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing outputs instead of overwriting. Requires --resume-jsonl (or an explicit --output path that exists).",
    )
    ap.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Per-call retries inside llm.call_once_stock (default 6). Increase if you hit transient rate limits.",
    )
    ap.add_argument(
        "--heartbeat-every",
        type=float,
        default=10.0,
        help="While waiting on the LLM HTTP call, print a heartbeat line every N seconds (default 10).",
    )
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip rows with missing required indicator fields (avoids early-window nulls).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Optional row limit (0 = all).")
    ap.add_argument("--where-ticker", default=None, help="Optional filter, e.g. MSFT.")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N calls (default 10). Use 1 for very chatty output.",
    )
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
    if args.all_variants and args.resume and args.resume_jsonl and not args.output:
        ap.error("--resume-jsonl with --all-variants is ambiguous (each variant writes a different file). "
                 "Re-run with --variant <n> pointing at that variant's JSONL, or pass an explicit per-run --output scheme.")
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

        # --- run ticker-by-ticker for visibility and easier debugging ---
        if args.where_ticker:
            tickers = [str(args.where_ticker)]
        else:
            tickers = sorted({str(r.get("Ticker")) for r in rows if r.get("Ticker") is not None})

        if args.resume and not args.resume_jsonl and not args.output:
            ap.error("--resume requires --resume-jsonl (or an explicit existing --output path).")

        if args.output:
            out_path = str(args.output)
        elif args.resume and args.resume_jsonl:
            out_path = str(args.resume_jsonl)
        else:
            out_path = _default_output_path_v2(
                provider=provider,
                variant=int(variant),
                mode=str(args.mode),
                input_path=str(args.input),
                ticker=str(args.where_ticker) if args.where_ticker else None,
                window_by=str(args.window_by),
                cal_span=int(cal_span),
                cal_step=int(cal_step),
                trading_n=int(args.window_trading_days),
                trading_step=int(step_trading),
            )

        # Precompute per-ticker work units so we can show useful progress.
        work_per_ticker: dict[str, list[Any]] = {}
        if args.mode == "window":
            for t in tickers:
                trows = [r for r in rows if str(r.get("Ticker")) == t]
                if args.window_by == "calendar":
                    chunks = _chunk_rows_calendar(
                        trows,
                        span_calendar_days=cal_span,
                        step_calendar_days=cal_step,
                        min_trading_rows=int(args.min_window_days),
                    )
                else:
                    chunks = _chunk_rows(
                        trows,
                        window_days=int(args.window_trading_days),
                        step_days=step_trading,
                        min_window_days=int(args.min_window_days),
                    )
                if limit_n:
                    chunks = chunks[:limit_n]
                work_per_ticker[t] = chunks
            total = sum(len(vv) for vv in work_per_ticker.values())
        else:
            for t in tickers:
                trows = [r for r in rows if str(r.get("Ticker")) == t]
                if limit_n:
                    trows = trows[:limit_n]
                work_per_ticker[t] = trows
            total = sum(len(vv) for vv in work_per_ticker.values())

        out_csv_path = args.output_csv or _default_output_csv_path(out_path)

        resume_global_done = 0
        resume_ticker: Optional[str] = None
        resume_local_completed = 0
        if args.resume:
            if not os.path.exists(out_path):
                ap.error(f"--resume specified but output JSONL does not exist: {out_path}")
            resume_global_done, resume_ticker, resume_local_completed, _seen_ok = _infer_resume_state_from_jsonl(
                out_path,
                expected_provider=str(provider),
                expected_variant=int(variant),
                work_per_ticker=work_per_ticker,
                mode=str(args.mode),
            )
            if resume_ticker is None:
                print(f"v{variant}: resume found nothing to do (already complete).", flush=True)
                continue

        ok = 0
        file_mode = "a" if args.resume else "w"
        csv_exists = os.path.exists(out_csv_path) and os.path.getsize(out_csv_path) > 0

        with open(out_path, file_mode, encoding="utf-8") as f, open(out_csv_path, file_mode, encoding="utf-8", newline="") as cf:
            cw = csv.DictWriter(cf, fieldnames=csv_fieldnames)
            if file_mode == "w" or not csv_exists:
                cw.writeheader()

            done = int(resume_global_done)
            if args.resume:
                try:
                    with open(out_path, "r", encoding="utf-8") as rf:
                        for line in rf:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if str(obj.get("provider")) != str(provider):
                                continue
                            if int(obj.get("variant") or -1) != int(variant):
                                continue
                            if obj.get("parse_error") is None and obj.get("parsed") is not None:
                                ok += 1
                except Exception:
                    ok = 0

            for t in tickers:
                units = work_per_ticker.get(t, [])
                planned_units = len(units)
                resume_base = 0
                if args.resume and resume_ticker:
                    if t < resume_ticker:
                        continue
                    if t == resume_ticker and resume_local_completed > 0:
                        resume_base = int(resume_local_completed)
                        units = units[int(resume_local_completed) :]

                print(
                    f"v{variant} {args.mode}: ticker={t} | units={len(units)}/{planned_units} | output={out_path} | resume={bool(args.resume)}",
                    flush=True,
                )

                if args.mode == "window":
                    for local_idx, chunk in enumerate(units):
                        abs_i0 = int(resume_base) + int(local_idx) + 1
                        print(
                            f"v{variant} window: calling_llm ticker={t} {abs_i0}/{planned_units} | "
                            f"rows={len(chunk)} | window={chunk[0].get('Date')}..{chunk[-1].get('Date')}",
                            flush=True,
                        )
                        hb_every = float(args.heartbeat_every) if float(args.heartbeat_every) > 0 else 10.0
                        hb_label = (
                            f"v{variant} window: waiting_on_llm ticker={t} {abs_i0}/{planned_units} | "
                            f"rows={len(chunk)}"
                        )
                        res = _with_wait_heartbeat(
                            hb_label,
                            hb_every,
                            lambda: call_window_once(
                                llm=llm,
                                rows=chunk,
                                variant=variant,
                                max_retries=int(args.max_retries),
                            ),
                        )
                        w_start = str(chunk[0].get("Date"))
                        w_end = str(chunk[-1].get("Date"))
                        rep_row = {"Ticker": chunk[0].get("Ticker"), "Date": w_end}

                        record: Dict[str, Any] = {
                            "provider": provider,
                            "variant": int(variant),
                            "row_index": int(done),
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
                        f.flush()
                        cw.writerow(
                            _flatten_for_csv(
                                provider,
                                int(variant),
                                int(done),
                                rep_row,
                                res.parsed,
                                res.parse_error,
                                res.raw_text,
                                window_start=w_start,
                                window_end=w_end,
                                window_n=len(chunk),
                            )
                        )
                        cf.flush()

                        done += 1
                        _maybe_rate_limit_cooldown(res.parse_error)
                        abs_i = int(resume_base) + int(local_idx) + 1
                        if args.progress_every > 0 and ((local_idx + 1) % int(args.progress_every) == 0 or done == total):
                            print(
                                f"v{variant} window: {done}/{total} done | ticker={t} {abs_i}/{planned_units} | parsed_ok={ok}",
                                flush=True,
                            )
                else:
                    for local_idx, row in enumerate(units):
                        abs_i0 = int(resume_base) + int(local_idx) + 1
                        print(
                            f"v{variant}: calling_llm ticker={t} {abs_i0}/{planned_units} | date={row.get('Date')}",
                            flush=True,
                        )
                        hb_every = float(args.heartbeat_every) if float(args.heartbeat_every) > 0 else 10.0
                        hb_label = f"v{variant}: waiting_on_llm ticker={t} {abs_i0}/{planned_units}"
                        res = _with_wait_heartbeat(
                            hb_label,
                            hb_every,
                            lambda: call_stock_once(
                                llm=llm,
                                row=row,
                                variant=variant,
                                max_retries=int(args.max_retries),
                            ),
                        )

                        record = {
                            "provider": provider,
                            "variant": int(variant),
                            "row_index": int(done),
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
                        f.flush()
                        cw.writerow(_flatten_for_csv(provider, int(variant), int(done), row, res.parsed, res.parse_error, res.raw_text))
                        cf.flush()

                        done += 1
                        _maybe_rate_limit_cooldown(res.parse_error)
                        abs_i = int(resume_base) + int(local_idx) + 1
                        if args.progress_every > 0 and ((local_idx + 1) % int(args.progress_every) == 0 or done == total):
                            print(
                                f"v{variant}: {done}/{total} done | ticker={t} {abs_i}/{planned_units} | parsed_ok={ok}",
                                flush=True,
                            )

                # once we finish the resumed ticker once, don't keep slicing for later variants
                if args.resume and resume_ticker and t == resume_ticker:
                    resume_local_completed = 0
                    resume_ticker = None

        print(f"v{variant}: wrote {total} records to {out_path} (parsed_ok={ok}).")
        print(f"v{variant}: wrote flattened CSV to {out_csv_path}.")


if __name__ == "__main__":
    main()

