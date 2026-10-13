#!/usr/bin/env python3
"""Nestor probe: validate Kronos inference on Apple Silicon and measure latency.

Fetches the last 400 15-minute candles for a Binance USDT-M symbol, loads a
Kronos model tier, and times KronosPredictor.predict for sample_count=1 and
sample_count=50 (pred_len=16, T=1.0, top_p=0.9).

API note: KronosPredictor.predict always averages the sample_count samples into
a single mean path (np.mean inside model/kronos.py:auto_regressive_inference).
For the shadow-forecaster role we need the raw per-sample paths, so this probe
also runs sample_paths(), a copy of that inference loop without the averaging.

Usage:
    python scripts/nestor_probe.py --model small --symbol BTCUSDT
    python scripts/nestor_probe.py --model mini  --symbol BTCUSDT --samples 50

Tip: after the first download, run with HF_HUB_OFFLINE=1 — the tokenizer's
online cache revalidation costs ~80s per process start; offline mode loads in <1s.
"""

import argparse
import json
import resource
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from model.kronos import (  # noqa: E402
    Kronos,
    KronosPredictor,
    KronosTokenizer,
    calc_time_stamps,
    sample_from_logits,
)

MODEL_CONFIGS = {
    "mini": {
        "model": "NeoQuasar/Kronos-mini",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-2k",
        "max_context": 2048,
    },
    "small": {
        "model": "NeoQuasar/Kronos-small",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
    "base": {
        "model": "NeoQuasar/Kronos-base",
        "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
        "max_context": 512,
    },
}

PRICE_COLS = ["open", "high", "low", "close"]
ALL_COLS = PRICE_COLS + ["volume", "amount"]


def rss_mb():
    """Peak resident set size of this process in MB (macOS ru_maxrss is bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def fetch_klines(symbol, limit, base_url):
    url = f"{base_url}?symbol={symbol}&interval=15m&limit={limit}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = json.load(resp)
    df = pd.DataFrame(
        {
            "open": [float(k[1]) for k in raw],
            "high": [float(k[2]) for k in raw],
            "low": [float(k[3]) for k in raw],
            "close": [float(k[4]) for k in raw],
            "volume": [float(k[5]) for k in raw],
            "amount": [float(k[7]) for k in raw],  # quote asset volume
        }
    )
    df["timestamps"] = pd.to_datetime([k[0] for k in raw], unit="ms")
    return df


def sample_paths(predictor, df, x_timestamp, y_timestamp, pred_len, T, top_p, sample_count):
    """Copy of model/kronos.py:auto_regressive_inference WITHOUT the np.mean
    over samples. Returns per-sample paths of shape [sample_count, pred_len, 6],
    de-normalized back to price scale."""
    tokenizer, model = predictor.tokenizer, predictor.model
    max_context, clip = predictor.max_context, predictor.clip
    device = predictor.device

    with torch.no_grad():
        x = df[ALL_COLS].values.astype(np.float32)
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
        z = tokenizer.decode(input_tokens, half=True)  # [sample_count, window, 6]
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        paths = z[:, :, -pred_len:, :].cpu().numpy().squeeze(0)  # [sample_count, pred_len, 6]
        paths = paths * (x_std + 1e-5) + x_mean
        return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=sorted(MODEL_CONFIGS))
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--samples", default="1,50", help="comma list of sample_count values to time")
    ap.add_argument("--pred-len", type=int, default=16)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--klines-url", default="https://api.binance.com/api/v3/klines")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    sample_counts = [int(s) for s in args.samples.split(",")]

    print(f"=== nestor probe: model={args.model} symbol={args.symbol} ===")
    print(f"python {sys.version.split()[0]} | torch {torch.__version__} | "
          f"mps_available={torch.backends.mps.is_available()}")
    print(f"peak RSS at start: {rss_mb():.0f} MB")

    df = fetch_klines(args.symbol, args.limit, args.klines_url)
    x_df = df[ALL_COLS]
    x_timestamp = df["timestamps"]
    last_ts = x_timestamp.iloc[-1]
    y_timestamp = pd.Series(pd.date_range(start=last_ts + pd.Timedelta(minutes=15),
                                          periods=args.pred_len, freq="15min"))
    print(f"fetched {len(df)} 15m candles | last open time {last_ts} | last real close {df['close'].iloc[-1]:.2f}")

    t0 = time.perf_counter()
    tokenizer = KronosTokenizer.from_pretrained(cfg["tokenizer"])
    model = Kronos.from_pretrained(cfg["model"])
    # Required on Apple Silicon: KronosPredictor never calls .eval(), leaving the
    # model in train mode, which makes MPS SDPA fail ("does not support dropout")
    # and would also randomly perturb forecasts via dropout at inference.
    model.eval()
    tokenizer.eval()
    load_s = time.perf_counter() - t0
    print(f"model+tokenizer load: {load_s:.1f}s | peak RSS {rss_mb():.0f} MB")

    predictor = KronosPredictor(model, tokenizer,
                                device=None if args.device == "auto" else args.device,
                                max_context=cfg["max_context"])
    print(f"device: {predictor.device} | max_context: {predictor.max_context}")

    for sc in sample_counts:
        torch.mps.empty_cache() if predictor.device == "mps" else None
        t0 = time.perf_counter()
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=args.pred_len, T=1.0, top_p=0.9, sample_count=sc, verbose=False,
        )
        dt = time.perf_counter() - t0
        print(f"\n[predict sample_count={sc}] wall {dt:.2f}s | peak RSS {rss_mb():.0f} MB")
        print("pred_df head:")
        print(pred_df.head().to_string(float_format=lambda v: f"{v:,.2f}"))
        fc = pred_df["close"].values
        print(f"sanity: last real close {df['close'].iloc[-1]:.2f} | "
              f"first forecast close {fc[0]:.2f} | last forecast close {fc[-1]:.2f}")

        if sc > 1:
            torch.mps.empty_cache() if predictor.device == "mps" else None
            t0 = time.perf_counter()
            paths = sample_paths(predictor, x_df, x_timestamp, y_timestamp,
                                 args.pred_len, 1.0, 0.9, sc)
            dt2 = time.perf_counter() - t0
            fc_paths = paths[:, :, 3]  # close column
            print(f"\n[raw paths sample_count={sc}] wall {dt2:.2f}s | peak RSS {rss_mb():.0f} MB")
            print(f"paths shape: {paths.shape} (samples, pred_len, vars)")
            print(f"first-step close across paths: min {fc_paths[:,0].min():,.2f} | "
                  f"mean {fc_paths[:,0].mean():,.2f} | max {fc_paths[:,0].max():,.2f} | "
                  f"std {fc_paths[:,0].std():,.2f}")
            print(f"last-step  close across paths: min {fc_paths[:,-1].min():,.2f} | "
                  f"mean {fc_paths[:,-1].mean():,.2f} | max {fc_paths[:,-1].max():,.2f} | "
                  f"std {fc_paths[:,-1].std():,.2f}")

    print(f"\ntotal peak RSS: {rss_mb():.0f} MB")


if __name__ == "__main__":
    main()
