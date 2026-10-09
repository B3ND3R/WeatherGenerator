# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Tests for DataReaderObs on small synthetic obs stores held in memory.

The stores follow the layout of the observation zarrs: a `data` table sorted by time, a
`dates` column, and an hourly index `idx_<base yyyymmddhhmm>_1` where entry h is the number
of rows with date <= base + h hours. Every test compares against a brute-force oracle: the
rows with t_start <= date < t_end.
"""

import numpy as np
import pytest
import zarr
from numpy.typing import NDArray
from zarr.storage import MemoryStore

from weathergen.datasets.data_reader_base import DTRange, TimeWindowHandler
from weathergen.datasets.data_reader_obs import DataReaderObs

ONE_HOUR = np.timedelta64(1, "h")
BASE = np.datetime64("1970-01-01T00:00", "ns")
COLNAMES = ["lat", "lon", "stalt", "obsvalue_tp_0"]
VALUE_COL = COLNAMES.index("obsvalue_tp_0")


def ts(s: str) -> np.datetime64:
    return np.datetime64(s, "ns")


def make_store(
    dates: NDArray,
    base: np.datetime64 = BASE,
    index_end: np.datetime64 | None = None,
) -> MemoryStore:
    """
    In-memory obs store with one row per entry of `dates`.

    obsvalue_tp_0 holds the row number, so returned rows can be identified. The hourly index
    is built like the real stores' (searchsorted, side="right") and runs to `index_end`
    (default: a day past the last date), so it can be made shorter than the data.
    """
    dates = np.asarray(dates, dtype="datetime64[ns]")
    n_rows = len(dates)
    data = np.zeros((n_rows, len(COLNAMES)), np.float32)
    data[:, COLNAMES.index("lat")] = -20.0
    data[:, COLNAMES.index("lon")] = 35.0
    data[:, COLNAMES.index("stalt")] = 100.0
    data[:, VALUE_COL] = np.arange(n_rows)

    store = MemoryStore()
    group = zarr.open_group(store, mode="w", zarr_format=2)
    arr = group.create_array("data", shape=data.shape, dtype="f4")
    arr[:] = data
    arr.attrs.update(colnames=COLNAMES, means=[0.0] * len(COLNAMES), vars=[1.0] * len(COLNAMES))
    dts = group.create_array("dates", shape=(n_rows, 1), dtype="M8[ns]")
    dts[:] = dates[:, None]

    if index_end is None:
        index_end = dates.max() + 24 * ONE_HOUR
    n_hours = int((index_end - base) / ONE_HOUR) + 1
    hrly = np.searchsorted(dates, base + np.arange(n_hours) * ONE_HOUR, side="right")
    name = f"idx_{base.astype('datetime64[m]').item():%Y%m%d%H%M}_1"
    idx = group.create_array(name, shape=hrly.shape, dtype="i8")
    idx[:] = hrly
    return store


def make_reader(
    store: MemoryStore,
    t_start: str,
    t_end: str,
    len_hrs: int,
    step_hrs: int,
    base: np.datetime64 = BASE,
) -> DataReaderObs:
    tw = TimeWindowHandler(ts(t_start), ts(t_end), len_hrs * ONE_HOUR, step_hrs * ONE_HOUR)
    stream_info = {
        "name": "test_obs",
        "geoinfo_channels": ["stalt"],
        "base_datetime": str(base.astype("datetime64[s]")),
    }
    return DataReaderObs(tw, store, stream_info, "train")


def expected_rows(dates: NDArray, t0: np.datetime64, t1: np.datetime64) -> NDArray:
    return np.nonzero((dates >= t0) & (dates < t1))[0]


def returned_rows(reader: DataReaderObs, k: int) -> NDArray:
    return reader._get(k, reader.source_idx).data[:, 0].astype(int)


# Daily totals for 3 stations, all stamped exactly at 06Z: every row sits on an hour boundary.
DAILY_06Z = np.repeat(ts("2020-01-02T06:00") + np.arange(5) * 24 * ONE_HOUR, 3)


def synop_like_dates(seed: int = 0) -> NDArray:
    """Sorted minute-resolution dates over 3 days, with many rows exactly on the hour."""
    rng = np.random.default_rng(seed)
    start = ts("2020-01-01T00:00")
    minutes = rng.integers(0, 3 * 24 * 60, size=2000)
    on_hour = rng.integers(0, 3 * 24, size=500) * 60
    return np.sort(start + np.concatenate([minutes, on_hour]) * np.timedelta64(1, "m"))


# Times to probe: on the hour, between hours, exactly on data, before/after all data,
# and before base_datetime / past the end of the index.
PROBE_TIMES = [
    "1969-12-31T00:00",
    "1970-01-01T00:00",
    "2019-12-31T23:00",
    "2020-01-01T00:00",
    "2020-01-01T00:30",
    "2020-01-01T06:00",
    "2020-01-02T06:00",
    "2020-01-02T06:01",
    "2020-01-02T05:59",
    "2020-01-03T13:17",
    "2020-01-06T06:00",
    "2020-01-08T00:00",
    "2030-01-01T00:00",
]


@pytest.fixture(params=["daily_06z", "synop_like"])
def dates(request) -> NDArray:
    return DAILY_06Z if request.param == "daily_06z" else synop_like_dates()


def test_hourly_index_is_loaded_into_memory():
    reader = make_reader(make_store(DAILY_06Z), "2020-01-02T00", "2020-01-04T00", 24, 24)
    assert not isinstance(reader.hrly_index, zarr.Array)


@pytest.mark.parametrize("t", PROBE_TIMES)
def test_bracket_contains_first_row_at_or_after(dates, t):
    reader = make_reader(make_store(dates), "2020-01-01T00", "2020-01-04T00", 6, 6)
    lo, hi = reader._bracket(ts(t))
    first = np.searchsorted(dates, ts(t), side="left")
    assert 0 <= lo <= first <= hi <= len(dates)


def test_bracket_is_one_hour_bin():
    # The bracket narrows the search to the rows of one hour bin, not the whole store.
    reader = make_reader(make_store(synop_like_dates()), "2020-01-01T00", "2020-01-04T00", 6, 6)
    lo, hi = reader._bracket(ts("2020-01-02T06:00"))
    bin_dates = reader.dt[lo:hi, 0]
    assert (bin_dates > ts("2020-01-02T05:00")).all()
    assert (bin_dates <= ts("2020-01-02T06:00")).all()


@pytest.mark.parametrize("t", PROBE_TIMES)
def test_first_row_at_or_after(dates, t):
    reader = make_reader(make_store(dates), "2020-01-01T00", "2020-01-04T00", 6, 6)
    assert reader._first_row_at_or_after(ts(t)) == np.searchsorted(dates, ts(t), side="left")


@pytest.mark.parametrize("t", PROBE_TIMES)
def test_first_row_at_or_after_index_shorter_than_data(t):
    # Real stores have rows after the last indexed hour; they must still be found.
    dates = synop_like_dates()
    store = make_store(dates, index_end=ts("2020-01-02T00:00"))
    reader = make_reader(store, "2020-01-01T00", "2020-01-04T00", 6, 6)
    assert reader._first_row_at_or_after(ts(t)) == np.searchsorted(dates, ts(t), side="left")


@pytest.mark.parametrize("t", ["2020-01-01T00:00", "2020-01-01T00:01", "2020-01-01T01:00"])
def test_first_row_at_or_after_non_default_base(t):
    # Data starting exactly at base_datetime: hour 0 of the index already counts those rows.
    base = ts("2020-01-01T00:00")
    dates = np.sort(np.concatenate([np.repeat(base, 3), synop_like_dates()]))
    reader = make_reader(make_store(dates, base=base), "2020-01-01T00", "2020-01-04T00", 6, 6, base)
    assert reader._first_row_at_or_after(ts(t)) == np.searchsorted(dates, ts(t), side="left")


def test_row_range_is_exact(dates):
    reader = make_reader(make_store(dates), "2020-01-01T00", "2020-01-04T00", 6, 6)
    for t0, t1 in [
        ("2020-01-02T00:00", "2020-01-02T06:00"),
        ("2020-01-02T06:00", "2020-01-02T12:00"),
        ("2020-01-01T03:00", "2020-01-02T03:00"),
        ("2020-01-01T00:00", "2020-01-08T00:00"),
        ("2019-01-01T00:00", "2019-01-02T00:00"),
    ]:
        start_row, end_row = reader._row_range(DTRange(ts(t0), ts(t1)))
        np.testing.assert_array_equal(
            np.arange(start_row, end_row), expected_rows(dates, ts(t0), ts(t1))
        )


def test_row_range_tiles_back_to_back_windows(dates):
    # end_row of each window is start_row of the next: no gaps, no overlaps.
    reader = make_reader(make_store(dates), "2020-01-01T00", "2020-01-08T00", 6, 6)
    edges = ts("2020-01-01T00:00") + np.arange(29) * 6 * ONE_HOUR
    ranges = [reader._row_range(DTRange(a, b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]
    for (_, end_row), (start_row, _) in zip(ranges[:-1], ranges[1:], strict=True):
        assert end_row == start_row
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(dates)


@pytest.mark.parametrize(
    "t_start, t_end, len_hrs, step_hrs",
    [
        ("2020-01-01T00", "2020-01-07T00", 6, 6),  # every 06Z row on a window start (bug 1)
        ("2020-01-01T00", "2020-01-07T00", 24, 24),  # no row on t_end (bug 2)
        ("2020-01-01T03", "2020-01-07T03", 6, 6),  # windows offset from the data
        ("2020-01-01T00", "2020-01-07T00", 12, 6),  # overlapping windows
        ("2020-01-02T06", "2020-01-07T06", 24, 24),  # window 0 starts exactly on data
    ],
)
def test_get_matches_oracle(dates, t_start, t_end, len_hrs, step_hrs):
    reader = make_reader(make_store(dates), t_start, t_end, len_hrs, step_hrs)
    for k in range(reader.len):
        win = reader.time_window_handler.window(k)
        np.testing.assert_array_equal(
            returned_rows(reader, k), expected_rows(dates, win.start, win.end)
        )


@pytest.mark.parametrize("len_hrs", [6, 24])
def test_get_returns_every_row_exactly_once(dates, len_hrs):
    reader = make_reader(make_store(dates), "2020-01-01T00", "2020-01-08T00", len_hrs, len_hrs)
    got = np.concatenate([returned_rows(reader, k) for k in range(reader.len)])
    np.testing.assert_array_equal(got, np.arange(len(dates)))


def test_get_returns_all_fields_for_the_window():
    reader = make_reader(make_store(DAILY_06Z), "2020-01-02T00", "2020-01-04T00", 24, 24)
    rdata = reader._get(0, reader.source_idx)
    assert rdata.coords.shape == (3, 2)
    assert rdata.geoinfos.shape == (3, 1)
    assert rdata.data.shape == (3, 1)
    np.testing.assert_array_equal(rdata.datetimes, np.repeat(ts("2020-01-02T06:00"), 3))
    np.testing.assert_array_equal(rdata.coords, [[-20.0, 35.0]] * 3)


def test_get_window_without_data_is_empty():
    reader = make_reader(make_store(DAILY_06Z), "2020-01-01T00", "2020-01-02T00", 6, 6)
    for k in range(reader.len):
        rdata = reader._get(k, reader.source_idx)
        assert rdata.data.shape == (0, 1)
        assert rdata.geoinfos.shape == (0, 1)


def test_get_rows_past_end_of_index():
    dates = synop_like_dates()
    store = make_store(dates, index_end=ts("2020-01-02T00:00"))
    # 6h, not 24h windows: _setup_sample_index fails if the index ends before the first
    # window does (a separate issue, unrelated to which rows _get reads).
    reader = make_reader(store, "2020-01-01T00", "2020-01-04T00", 6, 6)
    got = np.concatenate([returned_rows(reader, k) for k in range(reader.len)])
    np.testing.assert_array_equal(got, np.arange(len(dates)))


def test_get_without_channels_is_empty():
    reader = make_reader(make_store(DAILY_06Z), "2020-01-02T00", "2020-01-04T00", 24, 24)
    assert reader._get(0, []).data.shape == (0, 0)


def test_get_unsorted_dates_raises():
    # Out-of-order rows end up in a slice they don't belong to; check_reader_data catches it.
    dates = DAILY_06Z.copy()
    dates[[0, -1]] = dates[[-1, 0]]
    reader = make_reader(make_store(dates), "2020-01-02T00", "2020-01-07T00", 24, 24)
    with pytest.raises(AssertionError, match="violate window"):
        for k in range(reader.len):
            reader._get(k, reader.source_idx)
