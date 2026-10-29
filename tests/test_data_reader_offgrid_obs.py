"""Tests for the observation targets of the offgrid reader (offgrid.obs)."""

import numpy as np
import pytest
import zarr

from weathergen.datasets.data_reader_base import DTRange
from weathergen.readers_extra.data_reader_offgrid import _ObsTargets

BASE = np.datetime64("1970-01-01T00:00", "ns")
ONE_HOUR = np.timedelta64(1, "h")
COLNAMES = ["seqno", "lat", "lon", "lsm", "obsvalue_tp_0"]


def _write_store(path, dates, rows):
    """Store in the layout of DataReaderObs, with the hourly index as in
    rainfall_stations_to_zarr.py: hrly_index[h] is the number of rows with date <= BASE + h."""
    dates = np.asarray(dates, dtype="datetime64[ns]")
    z = zarr.open_group(str(path), mode="w")
    data = z.create_array("data", shape=(len(rows), len(COLNAMES)), dtype="float32")
    data[:] = np.asarray(rows, dtype=np.float32)
    data.attrs["colnames"] = COLNAMES
    z.create_array("dates", shape=(len(dates), 1), dtype="datetime64[ns]")[:] = dates[:, None]
    n_hours = int(np.ceil((dates.max() - BASE) / ONE_HOUR)) + 2
    hours = BASE + np.arange(n_hours) * ONE_HOUR
    idx = np.searchsorted(dates, hours, side="right").astype(np.int64)
    z.create_array("idx_197001010000_1", shape=idx.shape, dtype="int64")[:] = idx
    return str(path)


def _day(d):
    start = np.datetime64(d, "ms")
    return DTRange(start, start + np.timedelta64(1, "D"))


@pytest.fixture
def store(tmp_path):
    # two days, with rows at 06:00 and 23:00; station coords in float32 like the gauge store
    dates = [
        "2023-06-01T06:00", "2023-06-01T23:00", "2023-06-01T23:00",
        "2023-06-02T06:00", "2023-06-02T23:00",
    ]  # fmt: skip
    rows = [
        [0, -16.22, 39.9, 0.5, 1.0],
        [0, -16.22, 39.9, 0.5, 2.0],
        [1, -24.8953094, 33.7490997, 0.9, 3.0],
        [0, -16.22, 39.9, 0.5, 4.0],
        [1, -24.8953094, 33.7490997, 0.9, 5.0],
    ]
    return _write_store(tmp_path / "obs.zarr", dates, rows)


def _obs(store, **cfg):
    return _ObsTargets(
        {"file": store, "channels": {"tp": "obsvalue_tp_0"}, **cfg}, ["tp", "t2m"], [], "CHIRPS"
    )


def test_rows_at_step_times_only(store):
    t = np.array([np.datetime64("2023-06-01T23:00", "ms")])
    rd = _obs(store).get(t, _day("2023-06-01"), ["tp"])

    np.testing.assert_array_equal(rd.data[:, 0], [2.0, 3.0])
    assert (rd.datetimes == np.datetime64("2023-06-01T23:00", "ns")).all()
    # each point has exactly the coordinates of its own row
    expected = np.array([[-16.22, 39.9], [-24.8953094, 33.7490997]], dtype=np.float32)
    np.testing.assert_array_equal(rd.coords, expected)


def test_all_rows_of_window(store):
    t = np.array([np.datetime64("2023-06-01T23:00", "ms")])
    rd = _obs(store, step_times_only=False).get(t, _day("2023-06-01"), ["tp"])

    np.testing.assert_array_equal(rd.data[:, 0], [1.0, 2.0, 3.0])


def test_no_rows_at_step_time_warns(store, caplog):
    t = np.array([np.datetime64("2023-06-01T12:00", "ms")])
    rd = _obs(store).get(t, _day("2023-06-01"), ["tp"])

    assert len(rd.datetimes) == 0 and rd.data.shape == (0, 1)
    assert "Check offgrid.time_offset" in caplog.text


def test_window_without_rows(store, caplog):
    t = np.array([np.datetime64("2023-06-05T23:00", "ms")])
    rd = _obs(store).get(t, _day("2023-06-05"), ["tp"])

    assert len(rd.datetimes) == 0
    assert "Check offgrid.time_offset" not in caplog.text


def test_unmapped_channel_is_nan(store):
    t = np.array([np.datetime64("2023-06-02T23:00", "ms")])
    rd = _obs(store).get(t, _day("2023-06-02"), ["tp", "t2m"])

    np.testing.assert_array_equal(rd.data[:, 0], [5.0])
    assert np.isnan(rd.data[:, 1]).all()


def test_static_geoinfos(store):
    obs = _ObsTargets(
        {"file": store, "channels": {"tp": "obsvalue_tp_0"}, "geoinfos": {"lsm": "lsm"}},
        ["tp"],
        ["lsm"],
        "ERA5",
    )
    t = np.array([np.datetime64("2023-06-01T23:00", "ms")])
    rd = obs.get(t, _day("2023-06-01"), ["tp"])

    np.testing.assert_allclose(rd.geoinfos[:, 0], [0.5, 0.9])


def test_config_errors(store):
    with pytest.raises(AssertionError, match="not target channels"):
        _ObsTargets({"file": store, "channels": {"q": "obsvalue_tp_0"}}, ["tp"], [], "S")
    with pytest.raises(AssertionError, match="not in offgrid.obs.geoinfos"):
        _ObsTargets({"file": store, "channels": {"tp": "obsvalue_tp_0"}}, ["tp"], ["lsm"], "S")
    with pytest.raises(AssertionError, match="are not in"):
        _ObsTargets({"file": store, "channels": {"tp": "obsvalue_t2m"}}, ["tp"], [], "S")
