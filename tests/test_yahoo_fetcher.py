"""Unit tests for the Yahoo 15m fetcher (sidecar/yahoo.py).

Fixture mirrors the Yahoo chart v8 JSON shape consumed by daytrader-agent's
server/marketData.ts: chart.result[0].timestamp (Unix seconds, bar OPEN time)
+ indicators.quote[0].{open,high,low,close,volume}.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sidecar"))

from yahoo import BAR_MS, bars_from_payload  # noqa: E402

# Fixed "now": 2026-10-13 20:00:00 UTC (market long closed; all fixture bars
# below except the explicitly future one are closed against this instant).


def _ts(y, mo, d, h, mi):
    from datetime import datetime, timezone
    return int(datetime(y, mo, d, h, mi, tzinfo=timezone.utc).timestamp())


NOW_MS = _ts(2026, 10, 13, 20, 0) * 1000

T0 = _ts(2026, 10, 13, 13, 30)  # 9:30 ET bar open (EDT)


def _fixture():
    # 5 bars: 4 well-formed closed bars, 1 with null close (skipped), plus a
    # still-forming bar at the end (open = now - 5m).
    ts = [T0, T0 + 900, T0 + 1800, T0 + 2700, T0 + 3600, NOW_MS // 1000 - 300]
    opens = [100.0, 101.0, 102.0, 103.0, None, 105.0]
    highs = [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]
    lows = [99.0, 100.0, 101.0, 102.0, 103.0, 104.0]
    closes = [100.5, 101.5, 102.5, 103.5, None, 105.5]
    volumes = [1000, 1100, 1200, None, 1300, 1400]
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": ts,
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": highs,
                                "low": lows,
                                "close": closes,
                                "volume": volumes,
                            }
                        ]
                    },
                }
            ],
        }
    }
    return payload, ts


def test_bars_from_payload_normalizes_shape():
    payload, ts = _fixture()
    df = bars_from_payload(payload, now_ms=NOW_MS)
    # null-close bar and still-forming bar dropped; volume-None bar kept with 0
    assert list(df.columns) == ["open_time_ms", "open", "high", "low", "close", "volume", "amount"]
    assert len(df) == 4
    assert df["open_time_ms"].tolist() == [t * 1000 for t in ts[:4]]
    assert df["close"].tolist() == [100.5, 101.5, 102.5, 103.5]
    # volume missing -> 0 (marketData.ts parity)
    assert df["volume"].iloc[3] == 0.0
    # amount approximated as dollar volume (close * volume)
    assert df["amount"].iloc[0] == pytest.approx(100.5 * 1000.0)


def test_bars_from_payload_drops_still_forming_bar():
    payload, ts = _fixture()
    df = bars_from_payload(payload, now_ms=NOW_MS)
    newest_open = int(df["open_time_ms"].iloc[-1])
    assert newest_open + BAR_MS <= NOW_MS
    # the forming bar (open = now - 5m) must not appear
    assert (NOW_MS // 1000 - 300) * 1000 not in df["open_time_ms"].tolist()


def test_bars_from_payload_skips_null_ohlc_bar():
    payload, _ = _fixture()
    df = bars_from_payload(payload, now_ms=NOW_MS)
    assert 102.5 in df["close"].tolist()
    # the None-close bar (open T0+3600) is gone entirely
    assert (T0 + 3600) * 1000 not in df["open_time_ms"].tolist()


def test_bar_time_is_open_time_iso_utc():
    payload, ts = _fixture()
    df = bars_from_payload(payload, now_ms=NOW_MS)
    from datetime import datetime, timezone
    bar_time = datetime.fromtimestamp(int(df["open_time_ms"].iloc[0]) / 1000, tz=timezone.utc).isoformat()
    assert bar_time == "2026-10-13T13:30:00+00:00"


def test_bars_from_payload_raises_on_yahoo_error():
    payload = {"chart": {"error": {"description": "No data found"}, "result": None}}
    with pytest.raises(RuntimeError, match="Yahoo error"):
        bars_from_payload(payload, now_ms=NOW_MS)


def test_bars_from_payload_raises_when_all_bars_unusable():
    payload, _ = _fixture()
    payload["chart"]["result"][0]["timestamp"] = []
    with pytest.raises(RuntimeError, match="no usable candles"):
        bars_from_payload(payload, now_ms=NOW_MS)


def test_bars_from_payload_rejects_non_serializable_gracefully():
    # ensure the parser never returns an empty frame silently
    with pytest.raises(RuntimeError):
        bars_from_payload({}, now_ms=NOW_MS)
