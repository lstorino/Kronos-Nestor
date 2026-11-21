# Kronos shadow-forecaster sidecar

Data capture runs every 15 minutes (UTC bar close +45s) for Nestor's active
watchlist and observing recruitment candidates. Each cycle refreshes that
universe from the authenticated internal endpoint; on failure it keeps the
previous symbol lists. Binance covers crypto and Yahoo Finance covers US
instruments. Kronos uses 50 sampled paths, pred_len=16, T=1.0, top_p=0.9, and a
400×15m lookback, then stores raw candles and forecast distributions in SQLite.
The sidecar has no execution route or broker credentials.

The crypto symbols run every cycle. The US symbols piggyback on the same
cycle but only run inference when a new closed 15m bar exists (progress gate
against the newest forecast bar per symbol) — US bars close at :30/:45/:00/:15
ET, so the UTC-quarter cycle sometimes sees them at cycle start, sometimes
mid-cycle; a 1-cycle delay is fine (the consumer reads latest-by-symbol).

## Files

- `daemon.py` — scheduler + capture + inference + DB writer; loads the model once
- `server.py` — read-only HTTP API (runs as a thread inside the daemon)
- `pathwalk.py` — shared first-touch path-walking math (daemon + server)
- `yahoo.py` — keyless Yahoo chart v8 fetcher for US symbols (same internal
  bar shape as the Binance path; `amount` approximated as close × volume)

## Run modes

**Daemon (normal):** loads model once, wakes at each :00/:15/:30/:45+45s UTC.

```bash
python sidecar/daemon.py            # long-running
```

**One-shot:** exactly one capture cycle (fetch → candles upsert → inference →
store), no HTTP server; for cron-style operation and container smoke tests.

```bash
python sidecar/daemon.py --once
```

**Container (see root Dockerfile / docker-compose.yml):**

```bash
docker build -t kronos-nestor:local .
docker run -d -p 8787:8787 -v kronos-data:/data kronos-nestor:local
```

Models are baked into the image; runtime uses `HF_HUB_OFFLINE=1` (sub-second
load). SQLite lives in the `/data` volume — never on an exFAT filesystem
(SQLite journals break there; WAL silently loses committed frames).

Note for this Mac's exFAT repo: `docker build .` fails (BuildKit xattr walk on
exFAT). Pipe a tar context instead:

```bash
tar --exclude='._*' --exclude='*/._*' --exclude='./.git' --exclude='./data' \
    -cf - . | docker build -t kronos-nestor:local -
```

amd64 builds happen natively on the target host (e.g. Synology Container
Manager) — no cross-build needed; the Dockerfile is architecture-neutral
(cpu-index torch wheel + multi-arch base).

## HTTP API (read-only)

- `GET /health` → `{"ok": true, "last_cycle": "<iso>", "symbols": <dynamic total>}` (watchlist + recruitment)
- `GET /latest?symbol=BTCUSDT` → newest forecast row (horizons + brackets,
  `paths_blob` omitted); `GET /latest` → all symbols.
- `POST /query` — first-touch probabilities of caller-supplied stop/target
  against the latest stored 50 paths (same conservative same-bar rule as the
  daemon's brackets: a bar touching both stop and target counts as stop-first):

```bash
curl -s -X POST localhost:8787/query -H 'Content-Type: application/json' \
  -d '{"symbol":"BTCUSDT","stop":83000,"target":84500}'
```

Response: `symbol, bar_time, last_close, entry, horizon, stop, target,
paths_age_sec, n_paths, p_target_first, p_stop_first, p_neither,
expected_pct_at_horizon, median_pct_at_horizon` (+`"stale": true` if the paths
are older than 20 min — the caller decides). `entry` defaults to `last_close`;
`horizon` defaults to 16 (max). Unknown symbol / no forecast → 404 JSON;
malformed body → 400.

## Env vars

| var | default | notes |
|---|---|---|
| `KRONOS_BINANCE_SYMBOLS` | `BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT` | fallback crypto symbols when the universe endpoint is unavailable |
| `KRONOS_YAHOO_SYMBOLS` | `NVDA,AAPL,QQQ,SPY` | fallback US symbols when the universe endpoint is unavailable |
| `KRONOS_FORECAST_UNIVERSE_URL` | derived from `KRONOS_PUSH_URL` | authenticated watchlist + recruitment universe refresh each cycle |
| `KRONOS_STOCK_DEFER_AFTER_S` | `600` | defer remaining stock work to the next cycle when the cycle is already this many seconds old (compute-envelope guard) |
| `KRONOS_DB_PATH` | `~/.local/share/nestor-kronos/kronos.db` (Mac) / `/data/kronos.db` (image) | SQLite location — keep on a journaling filesystem |
| `KRONOS_BIND_HOST` | `127.0.0.1` (`0.0.0.0` in the image) | HTTP bind host |
| `HF_HUB_OFFLINE` | unset (`1` in the image) | set after first model download |
| `KRONOS_PUSH_URL` / `KRONOS_PUSH_TOKEN` | unset (push disabled) | POST captured forecasts to the Nestor internal endpoint: crypto as a batch before stock inference, then each stock immediately after inference |

## Performance (measured)

- Apple Silicon (MPS): ~11s per symbol inference; full 5-symbol cycle ~57s
- Container CPU (arm64, Docker Desktop): ~47–67s per symbol; ~286s per cycle
- Model load: 0.3–0.4s offline; ~80s on first online load (hub revalidation)

## Deployment note

Bind to LAN/Tailscale only — **never port-forward this to the public
internet**. The VPS (Nestor) reaches this service over Tailscale or a reverse
tunnel (e.g. `ssh -R`). The API is unauthenticated read-only forecast data, but
it still maps your internal network topology.


## Tests

Use Python 3.12 and the pinned NumPy version to match the production-compatible pandas build:

```bash
uv venv /tmp/kronos-venv312 --python 3.12
uv pip install --python /tmp/kronos-venv312/bin/python -r requirements-dev.txt
/tmp/kronos-venv312/bin/python -m pytest -q tests
```
