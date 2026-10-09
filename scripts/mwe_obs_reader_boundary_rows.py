# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# ruff: noqa: T201

"""
Minimal working example: DataReaderObs drops rows at time-window boundaries.

For each window [t_start, t_end), the rows the reader returns are compared with the rows
whose timestamp actually falls in the window, read directly from the store's `dates`.

Bug 1 - rows stamped exactly at a window boundary are returned by neither window.
    `idx_<base>_1[h]` counts the rows with date <= base + h hours, so it points *past* rows
    stamped exactly at hour h. The window starting at h begins its slice at idx[h] and skips
    them (DataReaderObs._setup_sample_index, `indices_start`). The window ending at h reads
    them but masks them out, because its end is exclusive (DataReaderObs._get, `t_mask`).

Bug 2 - the last row before t_end is dropped.
    idx[h_end] is already an exclusive bound, but `indices_end = idx[h_end] - 1` and the slice
    `start_row:end_row` is exclusive too, so one row too many is removed. In synop this is
    hidden because rows stamped exactly at t_end exist and are masked out anyway. It shows up
    when no row sits at t_end, e.g. daily data stamped at 06Z with 24h windows from 00Z.

Part 1 runs on the real synop store and shows bug 1. Part 2 builds a tiny in-memory store of
daily rows (no files written) and shows both bugs.

Usage (read-only):
    .venv/bin/python scripts/mwe_obs_reader_boundary_rows.py [path/to/obs.zarr]
"""

import sys

import numpy as np
import zarr
from zarr.storage import MemoryStore

from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.datasets.data_reader_obs import DataReaderObs

SYNOP_STORE = (
    "/e/data1/slmet/ml_training/observations-ea-ofb-0001-1979-2025-combined-surface-v5.zarr"
)
ONE_HOUR = np.timedelta64(1, "h")
BASE = np.datetime64("1970-01-01T00", "ns")


def compare_windows(reader: DataReaderObs, tw: TimeWindowHandler, dates, hrly_index, n_windows):
    """Print expected vs returned row counts per non-empty window, and why rows are missing."""
    for k in range(n_windows):
        win = tw.window(k)
        # Ground truth: every row with t_start <= date < t_end. Rows of hour h lie in
        # [idx[h-1], idx[h]), so [idx[h0-1], idx[h1]) brackets the whole window.
        h0 = int((win.start - BASE) / ONE_HOUR)
        h1 = int((win.end - BASE) / ONE_HOUR)
        lo = int(hrly_index[h0 - 1]) if h0 > 0 else 0
        hi = int(hrly_index[min(h1, hrly_index.shape[0] - 1)])
        d = dates[lo:hi, 0]
        expected = np.nonzero((d >= win.start) & (d < win.end))[0] + lo

        returned = len(reader._get(k, reader.source_idx).datetimes)
        if len(expected) == 0 and returned == 0:
            continue
        start_row, end_row = int(reader.indices_start[k]), int(reader.indices_end[k])
        not_read = np.setdiff1d(expected, np.arange(start_row, end_row))
        on_start = np.sum(d[not_read - lo] == win.start)

        t0, t1 = win.start.astype("datetime64[m]"), win.end.astype("datetime64[m]")
        print(f"window {k}: [{t0}, {t1})")
        print(
            f"  rows in window {len(expected):>9,} | returned {returned:>9,} | "
            f"dropped {len(expected) - returned:>7,}"
        )
        print(f"    stamped exactly at t_start (bug 1): {on_start:,}")
        print(f"    last row before t_end      (bug 2): {len(not_read) - on_start:,}")


def part1_synop(path):
    print(f"=== Part 1: synop store, 6h windows at 00/06/12/18Z\n    {path}\n")
    tw = TimeWindowHandler(
        np.datetime64("2019-12-31T18", "ns"),
        np.datetime64("2020-01-01T18", "ns"),
        6 * ONE_HOUR,
        6 * ONE_HOUR,
    )
    reader = DataReaderObs(tw, path, {"name": "synop", "geoinfo_channels": ["lsm"]}, "train")
    z = zarr.open(path, mode="r")
    compare_windows(reader, tw, z["dates"], z["idx_197001010000_1"], n_windows=4)


def daily_store(n_stations=3, n_days=4):
    """In-memory obs store: one row per station per day, all stamped at 06Z.

    The hourly index is built the way the synop store's is: searchsorted(..., side="right").
    """
    cols = ["seqno", "lat", "lon", "obsvalue_tp_0", "stalt"]
    times = np.datetime64("2020-01-02T06", "ns") + np.arange(n_days) * 24 * ONE_HOUR
    dates = np.repeat(times, n_stations)
    data = np.zeros((len(dates), len(cols)), np.float32)
    data[:, 0] = np.tile(np.arange(n_stations), n_days)
    data[:, 1], data[:, 2] = -20.0, 35.0

    store = MemoryStore()
    group = zarr.open_group(store, mode="w", zarr_format=2)
    arr = group.create_array("data", shape=data.shape, dtype="f4")
    arr[:] = data
    arr.attrs.update(colnames=cols, means=[0.0] * len(cols), vars=[1.0] * len(cols))
    dts = group.create_array("dates", shape=(len(dates), 1), dtype="M8[ns]")
    dts[:] = dates[:, None]
    n_hours = int((dates.max() - BASE) / ONE_HOUR) + 48
    hrly = np.searchsorted(dates, BASE + np.arange(n_hours) * ONE_HOUR, side="right")
    idx = group.create_array("idx_197001010000_1", shape=hrly.shape, dtype="i8")
    idx[:] = hrly
    return store, group


def part2_daily():
    print("\n=== Part 2: in-memory store, 3 stations x 4 days, every row stamped at 06Z")
    store, group = daily_store()
    stream_info = {"name": "daily", "geoinfo_channels": ["stalt"]}
    for label, length in [("6h windows from 00Z", 6), ("24h windows from 00Z", 24)]:
        print(f"\n--- {label}")
        tw = TimeWindowHandler(
            np.datetime64("2020-01-02T00", "ns"),
            np.datetime64("2020-01-06T00", "ns"),
            length * ONE_HOUR,
            length * ONE_HOUR,
        )
        reader = DataReaderObs(tw, store, stream_info, "train")
        hrly_index = group["idx_197001010000_1"]
        compare_windows(reader, tw, group["dates"], hrly_index, n_windows=reader.len)


if __name__ == "__main__":
    part1_synop(sys.argv[1] if len(sys.argv) > 1 else SYNOP_STORE)
    part2_daily()
