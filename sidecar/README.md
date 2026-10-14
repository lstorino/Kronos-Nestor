# Kronos shadow-forecaster sidecar

SHADOW MODE data capture: every 15 minutes (UTC bar close +45s) it forecasts
BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT (Binance) and NVDA, AAPL, QQQ,
SPY (Yahoo Finance) with Kronos-small (50 sampled paths, pred_len=16, T=1.0,
top_p=0.9, 400×15m lookback), stores raw candles (future fine-tuning dataset)
and forecast distributions in SQLite. Zero trading influence.

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

- `GET /health` → `{"ok": true, "last_cycle": "<iso>", "symbols": 9}` (crypto + US total)
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
| `KRONOS_BINANCE_SYMBOLS` | `BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT` | comma-separated crypto symbols (Binance klines) |
| `KRONOS_YAHOO_SYMBOLS` | `NVDA,AAPL,QQQ,SPY` | comma-separated US symbols (Yahoo 15m bars) |
| `KRONOS_STOCK_DEFER_AFTER_S` | `600` | defer remaining stock work to the next cycle when the cycle is already this many seconds old (compute-envelope guard) |
| `KRONOS_DB_PATH` | `~/.local/share/nestor-kronos/kronos.db` (Mac) / `/data/kronos.db` (image) | SQLite location — keep on a journaling filesystem |
| `KRONOS_BIND_HOST` | `127.0.0.1` (`0.0.0.0` in the image) | HTTP bind host |
| `HF_HUB_OFFLINE` | unset (`1` in the image) | set after first model download |
| `KRONOS_PUSH_URL` / `KRONOS_PUSH_TOKEN` | unset (push disabled) | POST captured forecasts to the Nestor internal endpoint after each cycle |

## Performance (measured)

- Apple Silicon (MPS): ~11s per symbol inference; full 5-symbol cycle ~57s
- Container CPU (arm64, Docker Desktop): ~47–67s per symbol; ~286s per cycle
- Model load: 0.3–0.4s offline; ~80s on first online load (hub revalidation)

## Deployment note

Bind to LAN/Tailscale only — **never port-forward this to the public
internet**. The VPS (Nestor) reaches this service over Tailscale or a reverse
tunnel (e.g. `ssh -R`). The API is unauthenticated read-only forecast data, but
it still maps your internal network topology.
