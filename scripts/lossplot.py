#!/usr/bin/env python3
"""
plot_losses.py

Parse training logs like:
  ... Total Loss: 21.6221\tLocal DINO: 8.6409\tGlobal DINO: 2.1624\tiBOT: 5.4094\tLR: ...\tIteration:     0/60000 ...

and plot curves vs iteration.

Usage:
  python plot_losses.py /path/to/main_rank0.log -o plots
  python plot_losses.py /path/to/*.log -o plots --every 5 --smooth 50
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


ITER_RE = re.compile(r"Iteration:\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
# key: value (float) pattern, e.g. "Total Loss: 21.6221"
KV_RE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9 _/\-]*?):\s*(?P<val>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
# Matches common training log timestamps: 2024-01-15 14:23:45 or 2024-01-15T14:23:45
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")


def moving_average(y: List[float], w: int) -> List[float]:
    if w <= 1 or w > len(y):
        return y
    out = []
    s = 0.0
    for i, v in enumerate(y):
        s += v
        if i >= w:
            s -= y[i - w]
        if i >= w - 1:
            out.append(s / w)
        else:
            out.append(s / (i + 1))
    return out


def parse_log(path: Path) -> Tuple[List[int], Dict[str, List[float]], List[Optional[datetime]]]:
    iters: List[int] = []
    series: Dict[str, List[float]] = {}
    timestamps: List[Optional[datetime]] = []

    with path.open("r", errors="ignore") as f:
        for line in f:
            if "loss" not in line.lower():
                continue

            m_it = ITER_RE.search(line)
            if not m_it:
                continue
            it = int(m_it.group(1))

            # Try to parse a timestamp from the line
            m_ts = TS_RE.search(line)
            ts: Optional[datetime] = None
            if m_ts:
                try:
                    ts = datetime.fromisoformat(m_ts.group(1))
                except ValueError:
                    pass

            # Extract key-value floats
            kvs = {m.group("key").strip(): float(m.group("val")) for m in KV_RE.finditer(line)}

            # Heuristic: keep typical training scalars, ignore memory, rank, etc.
            # (You can remove this filter if you want everything.)
            drop_keys = {"Global Rank", "Memory"}
            kvs = {k: v for k, v in kvs.items() if k not in drop_keys}

            if not kvs:
                continue

            iters.append(it)
            timestamps.append(ts)
            for k, v in kvs.items():
                series.setdefault(k, []).append(v)

            # Ensure all keys have same length (pad missing keys with NaN)
            cur_len = len(iters)
            for k in list(series.keys()):
                if len(series[k]) < cur_len:
                    series[k].append(float("nan"))

    return iters, series, timestamps


def filter_by_hours(
    iters: List[int],
    series: Dict[str, List[float]],
    timestamps: List[Optional[datetime]],
    last_hours: float,
) -> Tuple[List[int], Dict[str, List[float]]]:
    """Keep only entries from the last `last_hours` hours based on log timestamps.
    Falls back to tail_percent=100 (no filtering) if timestamps are unavailable."""
    valid_ts = [ts for ts in timestamps if ts is not None]
    if not valid_ts:
        return iters, series

    cutoff = max(valid_ts) - timedelta(hours=last_hours)
    keep = [i for i, ts in enumerate(timestamps) if ts is not None and ts >= cutoff]
    if not keep:
        print(f"[WARN] No data found in the last {last_hours}h — showing all data instead.")
        return iters, series

    filtered_iters = [iters[i] for i in keep]
    filtered_series = {k: [v[i] for i in keep] for k, v in series.items()}
    return filtered_iters, filtered_series


def plot_series(
    iters: List[int],
    series: Dict[str, List[float]],
    out_dir: Path,
    prefix: str,
    every: int,
    smooth: int,
    tail_percent: float,
    include: List[str],
    exclude: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep only the latest tail_percent of the timeline (e.g., 50% = second half)
    tp = min(100.0, max(0.0, tail_percent))
    if tp < 100.0 and iters:
        start_idx = int(len(iters) * (1.0 - tp / 100.0))
        if start_idx >= len(iters):
            start_idx = len(iters) - 1
        iters = iters[start_idx:]
        series = {k: v[start_idx:] for k, v in series.items()}

    keys = list(series.keys())

    def keep(k: str) -> bool:
        lk = k.lower()
        if include and not any(s.lower() in lk for s in include):
            return False
        if exclude and any(s.lower() in lk for s in exclude):
            return False
        return True

    keys = [k for k in keys if keep(k)]
    if not keys:
        raise SystemExit("No metrics matched (check --include/--exclude).")

    # Downsample
    if every > 1:
        idx = list(range(0, len(iters), every))
        iters_ds = [iters[i] for i in idx]
    else:
        idx = None
        iters_ds = iters

    # One plot per metric (cleaner + avoids unreadable spaghetti)
    for k in keys:
        y = series[k]
        if idx is not None:
            y = [y[i] for i in idx]

        y_sm = moving_average(y, smooth)

        plt.figure()
        plt.plot(iters_ds, y, label="raw")
        if smooth > 1:
            plt.plot(iters_ds, y_sm, label=f"MA({smooth})")
        plt.xlabel("iteration")
        plt.ylabel(k)
        plt.title(f"{prefix} — {k}")
        plt.legend()
        plt.tight_layout()

        safe_k = re.sub(r"[^A-Za-z0-9._-]+", "_", k.strip()).strip("_")
        out_path = out_dir / f"{prefix}.{safe_k}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()

    # Also: an overview plot for the most common losses if present
    common = [kk for kk in ["Total Loss", "Local DINO", "Global DINO", "iBOT"] if kk in series and keep(kk)]
    if common:
        plt.figure()
        for kk in common:
            y = series[kk]
            if idx is not None:
                y = [y[i] for i in idx]
            y_sm = moving_average(y, smooth)
            plt.plot(iters_ds, y_sm if smooth > 1 else y, label=kk)
        plt.xlabel("iteration")
        plt.title(f"{prefix} — overview")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}.overview.png", dpi=150)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="log files (supports globs if your shell expands them)")
    ap.add_argument("-o", "--out", default="plots", help="output directory")
    ap.add_argument("--every", type=int, default=1, help="keep 1 point every N (downsample)")
    ap.add_argument("--smooth", type=int, default=1, help="moving average window (in plotted points)")
    ap.add_argument("--last-hours", type=float, default=6.0,
                    help="plot only entries from the last N hours (default: 6). "
                         "Set to 0 to disable time filtering and use --tail-percent instead.")
    ap.add_argument("--tail-percent", type=float, default=100.0, help="fallback: plot only the latest X%% of timeline (used only when --last-hours 0)")
    ap.add_argument("--include", nargs="*", default=[], help="keep metrics whose name contains any of these substrings")
    ap.add_argument("--exclude", nargs="*", default=["Teacher momentum", "Teacher temp"], help="drop metrics containing these substrings")
    args = ap.parse_args()

    out_dir = Path(args.out)

    for lp in args.logs:
        path = Path(lp)
        if not path.exists():
            raise SystemExit(f"Not found: {path}")

        iters, series, timestamps = parse_log(path)
        if not iters:
            print(f"[WARN] No loss lines parsed in {path}")
            continue

        if args.last_hours > 0:
            iters, series = filter_by_hours(iters, series, timestamps, args.last_hours)
            tail_percent = 100.0  # time filter already applied
        else:
            tail_percent = args.tail_percent

        prefix = path.stem
        plot_series(
            iters=iters,
            series=series,
            out_dir=out_dir,
            prefix=prefix,
            every=max(1, args.every),
            smooth=max(1, args.smooth),
            tail_percent=tail_percent,
            include=args.include,
            exclude=args.exclude,
        )
        print(f"[OK] {path} -> {out_dir}/ (parsed {len(iters)} steps, {len(series)} metrics)")


if __name__ == "__main__":
    main()          