#!/usr/bin/env python3
"""Read-only HTTP API for the Kronos shadow-forecaster sidecar.

Runs as a thread inside daemon.py (one launchd/container job). Stateless: reads
only the SQLite DB, never loads the model. Binds localhost by default; set
KRONOS_BIND_HOST=0.0.0.0 inside a container.

Endpoints:
  GET  /health              -> {"ok": true, "last_cycle": "<iso>", "symbols": 5}
  GET  /latest?symbol=XXX   -> newest forecast row for one symbol (paths_blob omitted)
  GET  /latest              -> newest forecast rows for all symbols
  POST /query               -> first-touch probabilities of caller-supplied
                               stop/target levels against the latest stored paths
"""

import io
import json
import logging
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for pathwalk import
from pathwalk import walk_paths  # noqa: E402

log = logging.getLogger("sidecar.server")

STALE_AFTER_SEC = 20 * 60
FORECAST_COLUMNS = [
    "id", "symbol", "bar_time", "last_close", "pred_len", "n_samples",
    "model", "created_at", "horizons_json", "brackets_json", "paths_blob",
]


def _row_to_json(row, columns):
    d = dict(zip(columns, row))
    d.pop("paths_blob", None)
    for key in ("horizons_json", "brackets_json"):
        raw = d.pop(key, None)
        if raw is not None:
            d[key.replace("_json", "")] = json.loads(raw)
    return d


def _latest_row(conn, symbol):
    return conn.execute(
        "SELECT * FROM forecasts WHERE symbol=? ORDER BY id DESC LIMIT 1", (symbol,)
    ).fetchone()


def compute_query(conn, body, max_horizon=16):
    """Shared logic for POST /query. Raises ValueError with a message for 4xx."""
    symbol = body.get("symbol")
    if not symbol or not isinstance(symbol, str):
        raise ValueError("missing 'symbol'")
    if "stop" not in body or "target" not in body:
        raise ValueError("missing 'stop' and/or 'target'")
    try:
        stop = float(body["stop"])
        target = float(body["target"])
        horizon = int(body.get("horizon", max_horizon))
    except (TypeError, ValueError):
        raise ValueError("stop/target/horizon must be numeric")
    if horizon < 1 or horizon > max_horizon:
        raise ValueError(f"horizon must be between 1 and {max_horizon}")
    if not stop < target:
        raise ValueError("stop must be below target")

    row = _latest_row(conn, symbol)
    if row is None:
        return None
    d = dict(zip(FORECAST_COLUMNS, row))
    last_close = float(d["last_close"])
    entry = float(body.get("entry", last_close))
    if entry <= 0:
        raise ValueError("entry must be positive")

    paths = np.load(io.BytesIO(d["paths_blob"]))["paths"]  # [n_paths, pred_len, 6]
    n_paths = paths.shape[0]
    res = walk_paths(paths, stop, target, n_bars=horizon)
    closes = paths[:, horizon - 1, 3]
    pct = (closes / entry - 1.0) * 100.0
    created_at = datetime.fromisoformat(d["created_at"])
    age_sec = (datetime.now(timezone.utc) - created_at).total_seconds()

    out = {
        "symbol": symbol,
        "bar_time": d["bar_time"],
        "last_close": last_close,
        "entry": entry,
        "horizon": horizon,
        "stop": stop,
        "target": target,
        "paths_age_sec": round(age_sec, 1),
        "n_paths": int(n_paths),
        "p_target_first": float(np.mean(res == "up")),
        "p_stop_first": float(np.mean(res == "down")),
        "p_neither": float(np.mean(res == "neither")),
        "expected_pct_at_horizon": float(np.mean(pct)),
        "median_pct_at_horizon": float(np.median(pct)),
    }
    if age_sec > STALE_AFTER_SEC:
        out["stale"] = True
    return out


def make_handler(db_path, state, max_horizon=16):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _db(self):
            return sqlite3.connect(str(db_path), timeout=10)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self._send(200, {
                    "ok": True,
                    "last_cycle": state.get("last_cycle"),
                    "symbols": state.get("n_symbols", 0),
                })
                return
            if parsed.path == "/latest":
                symbol = parse_qs(parsed.query).get("symbol", [None])[0]
                try:
                    conn = self._db()
                    try:
                        if symbol:
                            row = _latest_row(conn, symbol)
                            if row is None:
                                self._send(404, {"error": f"no forecast for {symbol}"})
                                return
                            self._send(200, _row_to_json(row, FORECAST_COLUMNS))
                        else:
                            rows = conn.execute(
                                "SELECT f.* FROM forecasts f "
                                "JOIN (SELECT symbol, MAX(id) AS mid FROM forecasts GROUP BY symbol) m "
                                "ON f.id = m.mid ORDER BY f.symbol"
                            ).fetchall()
                            self._send(200, {r[1]: _row_to_json(r, FORECAST_COLUMNS) for r in rows})
                    finally:
                        conn.close()
                except Exception as e:
                    log.exception("GET %s failed", self.path)
                    self._send(500, {"error": str(e)})
                return
            self._send(404, {"error": "not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/query":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode() or "{}")
            except Exception:
                self._send(400, {"error": "invalid JSON body"})
                return
            conn = None
            try:
                conn = self._db()
                result = compute_query(conn, body, max_horizon=max_horizon)
            except ValueError as e:
                self._send(400, {"error": str(e)})
                return
            except Exception as e:
                log.exception("POST /query failed")
                self._send(500, {"error": str(e)})
                return
            finally:
                if conn is not None:
                    conn.close()
            if result is None:
                self._send(404, {"error": f"no forecast for {body.get('symbol')}"})
                return
            self._send(200, result)

        def log_message(self, fmt, *args):
            log.debug("http: " + fmt, *args)

    return Handler


def start_server(db_path, state, host="127.0.0.1", port=8787):
    srv = ThreadingHTTPServer((host, port), make_handler(db_path, state))
    thread = threading.Thread(target=srv.serve_forever, daemon=True, name="http-server")
    thread.start()
    log.info("read-only HTTP server on %s:%d", host, port)
    return srv
