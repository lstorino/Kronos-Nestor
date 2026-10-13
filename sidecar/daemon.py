#!/usr/bin/env python3
"""Nestor Kronos shadow-forecaster sidecar daemon (SHADOW MODE: no trading).

Every 15 minutes, aligned to the UTC :00/:15/:30/:45 bar close +45s, for each of
the 5 symbols it:
  1. fetches the last 401 15m klines from Binance (public, keyless),
  2. upserts them into the `candles` table (INSERT OR REPLACE on symbol+open_time_ms),
  3. runs Kronos-small with 50 sampled paths (pred_len=16, T=1.0, top_p=0.9,
     lookback 400 closed bars) and stores horizon stats, bracket probabilities,
     and the raw [50,16,6] paths blob into the `forecasts` table.

A read-only HTTP server (sidecar/server.py) runs in a thread on 127.0.0.1:8787.

If KRONOS_PUSH_URL and KRONOS_PUSH_TOKEN are set, each cycle also pushes the
just-captured forecasts to the Nestor internal endpoint (POST
/api/internal/kronos-forecast, x-internal-token). Best-effort: push failures
are logged and never affect capture; the server upserts on (symbol, barTime)
so re-pushes are idempotent.

Run: python sidecar/daemon.py            (normal, launchd-managed)
     python sidecar/daemon.py --once      (single immediate cycle; for smoke tests)
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # must precede huggingface_hub import

import importlib
import importlib.util
import io
import json
import logging
import sqlite3
import sys
import time
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from model.kronos import (  # noqa: E402
    Kronos,
    KronosPredictor,
    KronosTokenizer,
    calc_time_stamps,
    sample_from_logits,
)
from pathwalk import walk_paths  # noqa: E402

# --- load the sibling server module by path (no package install, no sys.path games)
_spec = importlib.util.spec_from_file_location(
    "nestor_sidecar_server", Path(__file__).resolve().parent / "server.py"
)
server_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server_mod)

log = logging.getLogger("sidecar")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
DATA_DIR = REPO_ROOT / "data"
# The SQLite DB must NOT live on the exFAT SSD: SQLite's file journal cannot be
# rewritten there (every journal mode that writes a -journal file fails on the
# second commit, and WAL silently loses committed frames via the heap shm
# fallback). APFS internal-drive location; override with KRONOS_DB_PATH.
DB_PATH = Path(os.environ.get(
    "KRONOS_DB_PATH",
    Path.home() / ".local" / "share" / "nestor-kronos" / "kronos.db",
))
HTTP_PORT = 8787

# Model tier via KRONOS_MODEL (default small). Both tiers are baked into the
# image's HF cache at build time so offline use works for either.
MODEL_CONFIGS = {
    "NeoQuasar/Kronos-small": {"name": "kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base", "max_context": 512},
    "NeoQuasar/Kronos-mini": {"name": "kronos-mini", "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k", "max_context": 2048},
}
MODEL_REPO = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
if MODEL_REPO not in MODEL_CONFIGS:
    raise SystemExit(f"unknown KRONOS_MODEL {MODEL_REPO!r}; expected one of {sorted(MODEL_CONFIGS)}")
MODEL_NAME = MODEL_CONFIGS[MODEL_REPO]["name"]
TOKENIZER_REPO = MODEL_CONFIGS[MODEL_REPO]["tokenizer"]
MAX_CONTEXT = MODEL_CONFIGS[MODEL_REPO]["max_context"]
PRED_LEN = 16
N_SAMPLES = 50
T_TEMP = 1.0
TOP_P = 0.9
KLINE_LIMIT = 401  # one extra so dropping the still-forming bar leaves 400
KLINE_BASE = "https://api.binance.com/api/v3/klines"
ALL_COLS_FOR_INFER = ["open", "high", "low", "close", "volume", "amount"]

BRACKET_GRID = [[0.5, 0.25], [1.0, 0.5], [1.5, 0.75], [2.0, 1.0], [2.0, 2.0], [3.0, 1.5]]

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol TEXT NOT NULL,
    open_time_ms INTEGER NOT NULL,
    open REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL,
    PRIMARY KEY (symbol, open_time_ms)
);
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    last_close REAL,
    pred_len INTEGER,
    n_samples INTEGER,
    model TEXT,
    created_at TEXT,
    horizons_json TEXT,
    brackets_json TEXT,
    paths_blob BLOB
);
CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_id ON forecasts(symbol, id);
"""


# ---------------------------------------------------------------- klines

def fetch_klines(symbol):
    url = f"{KLINE_BASE}?symbol={symbol}&interval=15m&limit={KLINE_LIMIT}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = json.loads(resp.read().decode())
    return pd.DataFrame(
        {
            "open_time_ms": [int(k[0]) for k in raw],
            "open": [float(k[1]) for k in raw],
            "high": [float(k[2]) for k in raw],
            "low": [float(k[3]) for k in raw],
            "close": [float(k[4]) for k in raw],
            "volume": [float(k[5]) for k in raw],
            "amount": [float(k[7]) for k in raw],  # quote asset volume
        }
    )


def fetch_with_retry(symbol, attempts=2):
    for attempt in range(1, attempts + 1):
        try:
            return fetch_klines(symbol)
        except Exception:
            log.exception("fetch %s failed (attempt %d/%d)", symbol, attempt, attempts)
            if attempt < attempts:
                time.sleep(2)
    return None


# ---------------------------------------------------------------- model

def load_model():
    """Load once at startup. Offline-first; one online retry, then crash loudly."""
    try:
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_REPO)
        model = Kronos.from_pretrained(MODEL_REPO)
    except Exception:
        log.exception("offline load failed; retrying with HF_HUB_OFFLINE unset")
        os.environ.pop("HF_HUB_OFFLINE", None)
        import huggingface_hub.constants as hfc

        importlib.reload(hfc)
        tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_REPO)
        model = Kronos.from_pretrained(MODEL_REPO)  # raises -> crash loudly
    # REQUIRED on Apple Silicon: predictor never calls .eval(); train mode makes
    # MPS SDPA throw "does not support dropout" and adds dropout noise to forecasts.
    model.eval()
    tokenizer.eval()
    return model, tokenizer


def sample_paths(predictor, df, x_timestamp, y_timestamp, pred_len, T, top_p, sample_count):
    """Vendored from scripts/nestor_probe.py (which vendored
    model/kronos.py:auto_regressive_inference with the per-sample averaging removed).
    Returns de-normalized paths of shape [sample_count, pred_len, 6]."""
    tokenizer, model = predictor.tokenizer, predictor.model
    max_context, clip = predictor.max_context, predictor.clip
    device = predictor.device

    with torch.no_grad():
        x = df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = np.clip((x - x_mean) / (x_std + 1e-5), -clip, clip)

        x_t = torch.from_numpy(x).unsqueeze(0).repeat(sample_count, 1, 1).to(device)
        x_stamp_t = torch.from_numpy(calc_time_stamps(x_timestamp).values.astype(np.float32)).unsqueeze(0).repeat(sample_count, 1, 1).to(device)
        y_stamp_t = torch.from_numpy(calc_time_stamps(y_timestamp).values.astype(np.float32)).unsqueeze(0).repeat(sample_count, 1, 1).to(device)

        x_token = tokenizer.encode(x_t, half=True)
        initial_seq_len = x_t.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp_t, y_stamp_t], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)
        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        for i in range(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)
            if current_seq_len <= max_context:
                input_tokens = [pre_buffer[:, :window_len], post_buffer[:, :window_len]]
            else:
                input_tokens = [pre_buffer, post_buffer]
            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=0, top_p=top_p, sample_logits=True)
            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=0, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)
        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        paths = z[:, :, -pred_len:, :].cpu().numpy().squeeze(0)
        return paths * (x_std + 1e-5) + x_mean


# ---------------------------------------------------------------- stats

def horizon_stats(paths, last_close):
    """Per-horizon (1..pred_len) stats of the close over the sampled paths."""
    closes = paths[:, :, 3]  # [n_samples, pred_len]
    out = []
    for h in range(closes.shape[1]):
        c = closes[:, h]
        out.append({
            "h": h + 1,
            "mean": float(np.mean(c)),
            "std": float(np.std(c)),
            "p10": float(np.percentile(c, 10)),
            "p25": float(np.percentile(c, 25)),
            "p50": float(np.percentile(c, 50)),
            "p75": float(np.percentile(c, 75)),
            "p90": float(np.percentile(c, 90)),
            "p_up": float(np.mean(c > last_close)),
        })
    return out


def bracket_probs(paths, last_close):
    """Fraction of paths that hit +up_pct before -down_pct within pred_len bars
    (see pathwalk.walk_paths for the conservative same-bar rule)."""
    out = {}
    for up_pct, down_pct in BRACKET_GRID:
        up_level = last_close * (1 + up_pct / 100.0)
        down_level = last_close * (1 - down_pct / 100.0)
        res = walk_paths(paths, down_level, up_level)
        out[f"{up_pct}/{down_pct}"] = float(np.mean(res == "up"))
    return out


# ---------------------------------------------------------------- db

def open_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    # exFAT note: WAL/truncate/delete journals are only safe because DB_PATH is
    # on APFS. Do not move the DB back to the exFAT repo volume.
    conn.execute("PRAGMA journal_mode=TRUNCATE")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn


def upsert_candles(conn, symbol, df):
    rows = [(symbol, int(r.open_time_ms), r.open, r.high, r.low, r.close, r.volume, r.amount) for r in df.itertuples()]
    conn.executemany(
        "INSERT OR REPLACE INTO candles (symbol, open_time_ms, open, high, low, close, volume, amount) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def insert_forecast(conn, symbol, bar_time, last_close, horizons, brackets, paths):
    buf = io.BytesIO()
    np.savez_compressed(buf, paths=paths.astype(np.float32))
    conn.execute(
        "INSERT INTO forecasts (symbol, bar_time, last_close, pred_len, n_samples, model, created_at,"
        " horizons_json, brackets_json, paths_blob) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            symbol, bar_time, float(last_close), PRED_LEN, N_SAMPLES, MODEL_NAME,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(horizons), json.dumps(brackets), buf.getvalue(),
        ),
    )
    conn.commit()


def push_forecasts(payloads):
    """POST the just-captured forecasts to the Nestor internal endpoint
    (POST /api/internal/kronos-forecast). Best-effort by contract: failures are
    logged and NEVER crash or delay the next capture cycle — the SQLite rows
    are the source of truth and pushes are idempotent (upsert on the server)."""
    url = os.environ.get("KRONOS_PUSH_URL")
    token = os.environ.get("KRONOS_PUSH_TOKEN")
    if not url or not token or not payloads:
        return
    body = json.dumps({"forecasts": payloads}).encode()
    headers = {"Content-Type": "application/json", "x-internal-token": token}
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    log.info("push ok: %d forecasts accepted (attempt %d)", len(payloads), attempt)
                    return
                log.warning("push attempt %d: HTTP %d", attempt, resp.status)
        except Exception:
            log.exception("push attempt %d failed", attempt)
        if attempt < 3:
            time.sleep(5)
    log.error("push failed after 3 attempts (%d forecasts remain in SQLite only)", len(payloads))


# ---------------------------------------------------------------- cycle

def next_wake(after):
    """Next wake instant strictly after `after`, at bar_close + 45s on the
    UTC :00/:15/:30/:45 grid."""
    x = after - timedelta(minutes=15, seconds=45)
    floored = x.replace(minute=(x.minute // 15) * 15, second=0, microsecond=0)
    return floored + timedelta(minutes=30, seconds=45)


def run_cycle(predictor, conn, wake_dt):
    boundary_dt = wake_dt - timedelta(minutes=15, seconds=45)  # just-closed bar open time
    boundary_ms = int(boundary_dt.timestamp() * 1000)
    t_cycle = time.perf_counter()

    fetched = {}
    t_fetch = time.perf_counter()
    for sym in SYMBOLS:
        df = fetch_with_retry(sym)
        if df is None:
            log.error("%s: fetch failed twice, skipping this cycle", sym)
            continue
        upsert_candles(conn, sym, df)
        closed = df[df.open_time_ms <= boundary_ms]
        if len(closed) < 100:
            log.error("%s: only %d closed bars at boundary %s, skipping", sym, len(closed), boundary_dt)
            continue
        fetched[sym] = closed
    fetch_s = time.perf_counter() - t_fetch

    t_infer = time.perf_counter()
    payloads = []
    for sym, closed in fetched.items():
        t0 = time.perf_counter()
        try:
            x_timestamp = pd.to_datetime(closed["open_time_ms"], unit="ms")
            y_timestamp = pd.Series(pd.date_range(
                start=x_timestamp.iloc[-1] + pd.Timedelta(minutes=15),
                periods=PRED_LEN, freq="15min",
            ))
            paths = sample_paths(predictor, closed[ALL_COLS_FOR_INFER], x_timestamp,
                                 y_timestamp, PRED_LEN, T_TEMP, TOP_P, N_SAMPLES)
            last_close = float(closed["close"].iloc[-1])
            bar_time = datetime.fromtimestamp(int(closed["open_time_ms"].iloc[-1]) / 1000, tz=timezone.utc).isoformat()
            horizons = horizon_stats(paths, last_close)
            brackets = bracket_probs(paths, last_close)
            insert_forecast(conn, sym, bar_time, last_close, horizons, brackets, paths)
            payloads.append({
                "symbol": sym,
                "barTime": bar_time,
                "lastClose": last_close,
                "predLen": PRED_LEN,
                "nPaths": N_SAMPLES,
                "model": MODEL_NAME,
                "horizonsJson": horizons,
                "bracketsJson": brackets,
            })
            log.info("%s forecast ok bar=%s last_close=%.4f infer=%.1fs", sym, bar_time, last_close, time.perf_counter() - t0)
        except Exception:
            log.exception("%s: inference failed this cycle", sym)
        finally:
            if predictor.device == "mps":
                torch.mps.empty_cache()
    infer_s = time.perf_counter() - t_infer
    total_s = time.perf_counter() - t_cycle
    log.info("cycle boundary=%s fetched=%d/5 symbols fetch=%.1fs infer=%.1fs total=%.1fs",
             boundary_dt.isoformat(), len(fetched), fetch_s, infer_s, total_s)
    try:
        push_forecasts(payloads)
    except Exception:
        log.exception("push step crashed (ignored; capture unaffected)")
    return datetime.now(timezone.utc).isoformat()


def main():
    warnings.filterwarnings("ignore", message=".*generic.*unit.*", category=DeprecationWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
    )
    logging.Formatter.converter = time.gmtime  # UTC log timestamps
    bind_host = os.environ.get("KRONOS_BIND_HOST", "127.0.0.1")
    log.info("starting: symbols=%s model=%s n_samples=%d pred_len=%d host=%s port=%d db=%s",
             SYMBOLS, MODEL_NAME, N_SAMPLES, PRED_LEN, bind_host, HTTP_PORT, DB_PATH)

    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass  # not required at runtime (e.g. read-only image filesystem)
    conn = open_db()

    t0 = time.perf_counter()
    model, tokenizer = load_model()
    log.info("model+tokenizer loaded in %.1fs (device auto-detect)", time.perf_counter() - t0)
    predictor = KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)
    log.info("predictor ready on device=%s max_context=%d", predictor.device, predictor.max_context)

    if "--once" in sys.argv:
        # one-shot capture cycle: no HTTP server, run and exit (cron / container smoke)
        run_cycle(predictor, conn, datetime.now(timezone.utc))
        log.info("--once done, exiting")
        return

    state = {"last_cycle": None, "n_symbols": len(SYMBOLS)}
    server_mod.start_server(DB_PATH, state, host=bind_host, port=HTTP_PORT)

    while True:
        now = datetime.now(timezone.utc)
        wake = next_wake(now)
        log.info("sleeping until %s", wake.isoformat())
        time.sleep(max(0.0, (wake - now).total_seconds()))
        try:
            state["last_cycle"] = run_cycle(predictor, conn, wake)
        except Exception:
            log.exception("cycle failed; continuing to next boundary")
            state["last_cycle"] = datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
