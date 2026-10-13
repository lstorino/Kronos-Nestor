#!/usr/bin/env python3
"""Shared path-walking math for the Kronos shadow forecaster.

Used by both sidecar/daemon.py (bracket probabilities at capture time) and
sidecar/server.py (on-demand stop/target cross-checks at query time) so both
see identical first-touch semantics.
"""

import numpy as np


def walk_paths(paths, down_level, up_level, n_bars=None):
    """Classify each path by which level is touched first.

    paths: array [n_paths, n_steps, 6] (open, high, low, close, volume, amount).
    Walks each path bar-by-bar on its high/low, up to n_bars steps.

    Conservative same-bar rule: if a bar touches BOTH levels, it counts as
    'down' (we check the down level first). Paths that touch neither within
    n_bars are 'neither'.

    Returns a numpy object array of 'up' / 'down' / 'neither', one per path.
    """
    if n_bars is not None:
        paths = paths[:, :n_bars, :]
    highs = paths[:, :, 1]
    lows = paths[:, :, 2]
    results = np.full(paths.shape[0], "neither", dtype=object)
    for p in range(paths.shape[0]):
        for i in range(paths.shape[1]):
            if lows[p, i] <= down_level:
                results[p] = "down"
                break
            if highs[p, i] >= up_level:
                results[p] = "up"
                break
    return results
