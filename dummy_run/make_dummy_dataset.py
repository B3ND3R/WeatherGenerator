"""
Generates a synthetic (but spatially/temporally structured) anemoi-format zarr
dataset at real ERA5 1-degree resolution (360x180 regular lat/lon grid, ~65k
points) so that `uv run train` can be exercised end-to-end on a GPU without
access to real ERA5 data.

Fields are built from a handful of smooth travelling zonal waves plus a
latitude-dependent mean/seasonal cycle (rather than pure white noise), so
that maps, animations and other `evaluate` plots produced from a model
trained on this data look like plausible weather fields instead of static.

The output satisfies the minimal schema that `anemoi.datasets.open_dataset()`
reads (arrays: data/dates/latitudes/longitudes/mean/stdev/maximum/minimum,
plus attrs: resolution/frequency/variables/variables_metadata/constant_fields).

Usage:
    uv run python dummy_run/make_dummy_dataset.py
"""

from pathlib import Path

import numpy as np
import zarr

OUT_PATH = Path(__file__).parent / "data" / "dummy_era5_1deg.zarr"

VARIABLES = ["2t", "10u", "10v", "z_500", "t_850", "q_850"]
N_ENS = 1

# Regular 1-degree lat/lon grid, matching real ERA5 1deg resolution
# (offset off the poles to avoid degenerate longitude collapse there).
N_LAT, N_LON = 180, 360
N_POINTS = N_LAT * N_LON

START = np.datetime64("2020-01-01T00:00:00")
END = np.datetime64("2020-03-01T00:00:00")
FREQ_HOURS = 6

# Per-variable (mean_at_equator, pole_to_equator_delta, seasonal_amp, wave_amp, noise_amp)
# roughly in each variable's physical units, just enough structure to look plausible.
VAR_PARAMS = {
    "2t": {"mean": 298.0, "lat_delta": 55.0, "seasonal_amp": 12.0, "wave_amp": 4.0, "noise_amp": 1.0},
    "10u": {"mean": 0.0, "lat_delta": 0.0, "seasonal_amp": 2.0, "wave_amp": 8.0, "noise_amp": 1.5},
    "10v": {"mean": 0.0, "lat_delta": 0.0, "seasonal_amp": 1.0, "wave_amp": 5.0, "noise_amp": 1.5},
    "z_500": {"mean": 5500.0, "lat_delta": 1000.0, "seasonal_amp": 150.0, "wave_amp": 200.0, "noise_amp": 30.0},
    "t_850": {"mean": 285.0, "lat_delta": 45.0, "seasonal_amp": 10.0, "wave_amp": 3.0, "noise_amp": 1.0},
    "q_850": {"mean": 0.006, "lat_delta": 0.0055, "seasonal_amp": 0.001, "wave_amp": 0.0008, "noise_amp": 0.0003},
}


def _traveling_waves(lat_rad, lon_grid, t_days, rng, num_modes=4):
    """Sum of a few smooth eastward-travelling zonal waves, enveloped by latitude."""
    wave = np.zeros((t_days.shape[0], *lon_grid.shape), dtype=np.float64)
    envelope = np.cos(lat_rad) ** 1.5  # midlatitude/tropics envelope, ~0 at poles
    for mode in range(1, num_modes + 1):
        phase = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(0.15, 0.6) * (1 if mode % 2 == 0 else -1)  # rad/day, eastward/westward
        amp = 1.0 / mode
        wave += amp * np.cos(
            mode * lon_grid[None, :, :]
            + phase
            + speed * t_days[:, None, None]
        )
    return wave * envelope[None, :, :]


def main() -> None:
    rng = np.random.default_rng(0)

    lats1d = np.linspace(-89.5, 89.5, N_LAT)
    lons1d = np.linspace(-180, 180, N_LON, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lons1d, lats1d)
    latitudes = lat_grid.ravel().astype(np.float64)
    longitudes = lon_grid.ravel().astype(np.float64)
    assert latitudes.shape[0] == N_POINTS

    dates = np.arange(START, END, np.timedelta64(FREQ_HOURS, "h")).astype("datetime64[s]")
    n_time = dates.shape[0]
    n_vars = len(VARIABLES)

    day_of_year = (dates.astype("datetime64[D]") - dates.astype("datetime64[Y]")).astype(int)
    t_days = (dates - START) / np.timedelta64(1, "D")
    season_phase = 2 * np.pi * day_of_year / 365.25

    lat_rad = np.deg2rad(lat_grid)
    lon_rad = np.deg2rad(lon_grid)

    data = np.empty((n_time, n_vars, N_ENS, N_POINTS), dtype=np.float32)
    for vi, var in enumerate(VARIABLES):
        p = VAR_PARAMS[var]

        # Smooth latitude-dependent climatology (warm equator / cold poles).
        base = p["mean"] - p["lat_delta"] * np.sin(lat_rad) ** 2

        # Seasonal cycle, larger amplitude towards the poles, opposite phase per hemisphere.
        seasonal = (
            p["seasonal_amp"]
            * np.sign(lat_rad + 1e-9)
            * np.abs(np.sin(lat_rad))
            * np.cos(season_phase)[:, None, None]
        )

        wave = p["wave_amp"] * _traveling_waves(lat_rad, lon_rad, t_days, rng)

        # Small-scale spatially-smooth texture: a few higher-wavenumber, faster modes.
        texture = p["wave_amp"] * 0.3 * _traveling_waves(lat_rad, lon_rad, t_days * 3, rng, num_modes=8)

        noise = rng.normal(loc=0.0, scale=p["noise_amp"], size=(n_time, N_LAT, N_LON))

        field = base[None, :, :] + seasonal + wave + texture + noise
        if var == "q_850":
            field = np.clip(field, 0.0, None)

        data[:, vi, 0, :] = field.reshape(n_time, N_POINTS).astype(np.float32)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.open(str(OUT_PATH), mode="w")
    store.create_array("data", data=data, chunks=(1, n_vars, N_ENS, N_POINTS))
    store.create_array("dates", data=dates)
    store.create_array("latitudes", data=latitudes)
    store.create_array("longitudes", data=longitudes)
    store.create_array("mean", data=data.mean(axis=(0, 2, 3)))
    store.create_array("stdev", data=data.std(axis=(0, 2, 3)))
    store.create_array("maximum", data=data.max(axis=(0, 2, 3)))
    store.create_array("minimum", data=data.min(axis=(0, 2, 3)))

    store.attrs["resolution"] = "1deg"
    store.attrs["frequency"] = f"{FREQ_HOURS}h"
    store.attrs["variables"] = VARIABLES
    store.attrs["variables_metadata"] = {v: {} for v in VARIABLES}
    store.attrs["constant_fields"] = []
    store.attrs["missing_dates"] = []
    store.attrs["field_shape"] = [N_POINTS]

    print(f"Wrote synthetic 1-degree anemoi dataset to {OUT_PATH}")
    print(f"  grid: {N_LAT}x{N_LON} = {N_POINTS} points")
    print(f"  variables: {VARIABLES}")
    print(f"  time range: {dates[0]} .. {dates[-1]} ({n_time} steps @ {FREQ_HOURS}h)")


if __name__ == "__main__":
    main()
