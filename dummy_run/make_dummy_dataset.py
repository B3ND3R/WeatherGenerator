"""
Generates a tiny, fully synthetic (random) anemoi-format zarr dataset so that
`uv run train` can be exercised end-to-end without access to real ERA5 data.

The output satisfies the minimal schema that `anemoi.datasets.open_dataset()`
reads (arrays: data/dates/latitudes/longitudes/mean/stdev/maximum/minimum,
plus attrs: resolution/frequency/variables/variables_metadata/constant_fields).

Usage:
    uv run python dummy_run/make_dummy_dataset.py
"""

from pathlib import Path

import numpy as np
import zarr

# OUT_PATH = Path(__file__).parent / "data" / "dummy_era5.zarr"
OUT_PATH = Path("/Users/matt/data/dummy_era5.zarr")

VARIABLES = ["2t", "10u", "10v", "z_500", "t_850", "q_850"]
N_ENS = 1
N_POINTS = 2664  # ~5deg regular lat/lon grid

START = np.datetime64("2020-01-01T00:00:00")
END = np.datetime64("2020-03-01T00:00:00")
FREQ_HOURS = 6


def main() -> None:
    rng = np.random.default_rng(0)

    lats1d = np.linspace(-90, 90, 37)
    lons1d = np.linspace(-180, 180, 72, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons1d, lats1d)
    latitudes = lat_grid.ravel().astype(np.float64)
    longitudes = lon_grid.ravel().astype(np.float64)
    assert latitudes.shape[0] == N_POINTS

    dates = np.arange(START, END, np.timedelta64(FREQ_HOURS, "h")).astype("datetime64[s]")
    n_time = dates.shape[0]
    n_vars = len(VARIABLES)

    data = rng.normal(loc=0.0, scale=1.0, size=(n_time, n_vars, N_ENS, N_POINTS)).astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(OUT_PATH), mode="w")
    store.create_array("data", data=data, chunks=(1, n_vars, N_ENS, N_POINTS))
    store.create_array("dates", data=dates)
    store.create_array("latitudes", data=latitudes)
    store.create_array("longitudes", data=longitudes)
    store.create_array("mean", data=np.zeros(n_vars))
    store.create_array("stdev", data=np.ones(n_vars))
    store.create_array("maximum", data=np.full(n_vars, 5.0))
    store.create_array("minimum", data=np.full(n_vars, -5.0))

    store.attrs["resolution"] = "dummy"
    store.attrs["frequency"] = f"{FREQ_HOURS}h"
    store.attrs["variables"] = VARIABLES
    store.attrs["variables_metadata"] = {v: {} for v in VARIABLES}
    store.attrs["constant_fields"] = []
    store.attrs["missing_dates"] = []
    store.attrs["field_shape"] = [N_POINTS]

    print(f"Wrote synthetic anemoi dataset to {OUT_PATH}")
    print(f"  variables: {VARIABLES}")
    print(f"  time range: {dates[0]} .. {dates[-1]} ({n_time} steps @ {FREQ_HOURS}h)")


if __name__ == "__main__":
    main()
