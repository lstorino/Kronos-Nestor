#!/usr/bin/env python3
"""Yahoo Finance 15m bar fetcher for US symbols (keyless).

Mirrors daytrader-agent server/marketData.ts conventions: Yahoo chart v8 JSON
(`https://query1.finance.yahoo.com/v8/finance/chart/{SYMBOL}`), Unix timestamps
-> UTC, a bar is skipped when any OHLC value is missing (still-forming or
no-trade bar), volume defaults to 0 when absent, same User-Agent header.

Yahoo provides no quote-asset volume field, so `amount` (Binance kline field 7,
close*volume summed) is approximated as close * volume (dollar volume) to keep
the Kronos 6-column input shape identical to the Binance path.

Bar open times: Yahoo `timestamp[i]` is the bar OPEN time (same convention as
Binance k[0]). Downstream barTime is that open time as ISO8601 UTC.
"""

import contextlib
import json
import socket
import time
import urllib.request

import pandas as pd

YAHOO_CHART_BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_INTERVAL = "15m"
YAHOO_RANGE = "60d"  # 15m bars ~= 26 per US trading day; 60d yields 1000+ bars
BAR_MS = 15 * 60 * 1000
USER_AGENT = "Mozilla/5.0 (NESTOR trading agent; mock-mode observation)"

COLUMNS = ["open_time_ms", "open", "high", "low", "close", "volume", "amount"]


@contextlib.contextmanager
def _ipv4_preferred():
    """stdlib sockets try getaddrinfo results sequentially, so a blackholed
    IPv6 route burns the whole timeout before the IPv4 fallback is attempted
    (observed on the dev Mac: 2 dead v6 addrs x 15s per request). curl is fast
    only because it Happy-Eyeballs. Prefer IPv4 for the duration of one
    request; fall back to the default resolution when no A record exists."""
    orig = socket.getaddrinfo

    def ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
        try:
            return orig(host, port, socket.AF_INET, type, proto, flags)
        except socket.gaierror:
            return orig(host, port, family, type, proto, flags)

    socket.getaddrinfo = ipv4_first
    try:
        yield
    finally:
        socket.getaddrinfo = orig


def _num(quote, key, i):
    arr = quote.get(key)
    if not arr or i >= len(arr):
        return None
    v = arr[i]
    return float(v) if isinstance(v, (int, float)) else None


def bars_from_payload(payload, now_ms=None):
    """chart v8 JSON -> normalized closed-bar DataFrame.

    Columns: open_time_ms, open, high, low, close, volume, amount.
    Bars with a missing OHLC value and bars still forming at now_ms
    (bar open + 15m > now) are dropped. Raises RuntimeError on a Yahoo
    error envelope or when no usable candle remains.
    """
    chart = (payload or {}).get("chart")
    err = chart.get("error") if chart else None
    if err:
        raise RuntimeError(f"Yahoo error: {err.get('description') or 'unknown'}")
    if not chart:
        raise RuntimeError("Yahoo response contained no chart result")
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError("Yahoo response contained no chart result")
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    rows = []
    for i, ts in enumerate(timestamps):
        o = _num(quote, "open", i)
        h = _num(quote, "high", i)
        l = _num(quote, "low", i)
        c = _num(quote, "close", i)
        if o is None or h is None or l is None or c is None:
            continue  # still forming or no-trade bar (marketData.ts parity)
        open_ms = int(ts) * 1000
        if open_ms + BAR_MS > now_ms:
            continue  # still-forming bar has not closed yet
        v = _num(quote, "volume", i) or 0.0
        rows.append((open_ms, o, h, l, c, v, c * v))
    if not rows:
        raise RuntimeError("Yahoo response contained no usable candles")
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_yahoo_bars(symbol, attempts=2):
    """Fetch ~60d of 15m bars for a US symbol; returns the closed-bar frame."""
    url = f"{YAHOO_CHART_BASE}/{symbol}?interval={YAHOO_INTERVAL}&range={YAHOO_RANGE}"
    headers = {"User-Agent": USER_AGENT}
    last_err = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with _ipv4_preferred():
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return bars_from_payload(json.loads(resp.read().decode()))
        except Exception as e:  # noqa: BLE001 - retried by caller
            last_err = e
            if attempt < attempts:
                time.sleep(2)
    raise RuntimeError(f"Yahoo fetch {symbol} failed after {attempts} attempts: {last_err}")
