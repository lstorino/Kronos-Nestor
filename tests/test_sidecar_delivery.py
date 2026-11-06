import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sidecar"))

import daemon


class SidecarDeliveryTests(unittest.TestCase):
    def test_crypto_pushes_before_stock_inference_and_stock_pushes_immediately(self):
        wake = datetime(2026, 11, 6, 3, 15, 45, tzinfo=timezone.utc)
        boundary_ms = int(datetime(2026, 11, 6, 3, 0, tzinfo=timezone.utc).timestamp() * 1000)
        frame = pd.DataFrame({"open_time_ms": [boundary_ms - 400 * 900_000 + i * 900_000 for i in range(400)]})
        events = []

        def infer(_predictor, _conn, symbol, _closed, source, payloads):
            events.append(("infer", symbol, source))
            payloads.append({"symbol": symbol})

        def push(payloads):
            events.append(("push", [p["symbol"] for p in payloads]))

        with (
            patch.object(daemon, "SYMBOLS", ["BTCUSDT"]),
            patch.object(daemon, "YAHOO_SYMBOLS", ["QQQ"]),
            patch.object(daemon, "fetch_with_retry", return_value=frame),
            patch.object(daemon, "fetch_yahoo_bars", return_value=frame),
            patch.object(daemon, "upsert_candles"),
            patch.object(daemon, "latest_forecast_bar_ms", return_value=None),
            patch.object(daemon, "infer_symbol", side_effect=infer),
            patch.object(daemon, "push_forecasts", side_effect=push),
        ):
            daemon.run_cycle(None, None, wake)

        self.assertEqual(
            events,
            [
                ("infer", "BTCUSDT", "binance"),
                ("push", ["BTCUSDT"]),
                ("infer", "QQQ", "yahoo"),
                ("push", ["QQQ"]),
            ],
        )


if __name__ == "__main__":
    unittest.main()
