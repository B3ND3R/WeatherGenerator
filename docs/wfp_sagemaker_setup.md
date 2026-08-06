# SageMaker environment setup notes

Two issues block `uv run evaluate` (and likely other `uv run` entry points) in this
SageMaker Studio environment out of the box. Both are environment-specific, not bugs
in the WeatherGenerator repo itself.

## 1. `ModuleNotFoundError: No module named 'google.protobuf.json_format'`

**Cause:** `pyproject.toml` sets `[tool.uv] link-mode = "hardlink"` (previously
`"symlink"`). With `symlink` mode, `.venv/lib/.../site-packages` files are symlinks
into `~/.cache/uv`. In this SageMaker environment, `~/.cache` is wiped on instance
restart while `.venv` (which lives under the persistent `git/WeatherGenerator`
checkout) is not. That leaves thousands of dangling symlinks across the venv —
including `google/protobuf/json_format.py` — even though `protobuf` is correctly
declared as a dependency and `uv pip show protobuf` reports it as installed.

Check for dangling symlinks in the venv:
```bash
find .venv/lib/python3.12/site-packages -xtype l | wc -l
```

**Fix:** Force uv to relink/reinstall everything:
```bash
uv sync --reinstall
```

We also switched `link-mode` from `"symlink"` to `"hardlink"` in `pyproject.toml`.
Both `~/.cache/uv` and `.venv` live on the same filesystem here
(`/dev/nvme1n1`), so hardlinks are possible and survive a cache wipe (deleting one
directory entry doesn't free the underlying data while another hardlink still
references it). After changing `link-mode`, you must re-run `uv sync --reinstall`
once — editing the setting alone doesn't retroactively relink already-installed
packages.

Verify a file is now a real hardlink rather than a symlink:
```bash
python3 -c "
import os
p = '.venv/lib/python3.12/site-packages/google/protobuf/json_format.py'
print('is symlink:', os.path.islink(p))
print('nlink:', os.stat(p).st_nlink)
"
```

Note: if `~/.cache` is ever wiped again *while the repo/venv were also freshly
recreated* (e.g. a totally fresh clone), a normal `uv sync` will still populate
things correctly — this only bites when `.venv` survives a cache wipe.

## 2. `FileNotFoundError` / `AssertionError` around `WeatherGenerator-private`

**Cause:** Several modules load private/HPC configuration unconditionally at
**import time**, not just when their features are actually used:

- `packages/common/src/weathergen/common/paths.py::get_wg_private_path()` asserts
  that `<repo-root>/../WeatherGenerator-private` (or `$WEATHERGEN_PRIVATE_REPO_PATH`)
  exists.
- `packages/evaluate/src/weathergen/evaluate/plotting/plotter.py` reads
  `path_shared_working_dir` from the private config at module import time, just to
  set the Cartopy map-asset cache directory.
- `packages/evaluate/src/weathergen/evaluate/run_evaluation.py` calls
  `get_platform_env()` at module import time, which execs
  `<private-repo>/hpc/platform-env.py`.

`WeatherGenerator-private` is an internal GitLab repo
(`gitlab.jsc.fz-juelich.de/esde/WeatherGenerator-private`) used mainly for HPC
cluster paths, credentials, and MLflow config. It isn't available in this
SageMaker environment and isn't actually needed for a basic local `evaluate` run
(our `config/evaluate/eval_test.yml` sets `results_base_dir` explicitly per
run_id, so it doesn't depend on private path resolution for that part).

**Fix:** Stub out the private repo locally with just enough content to satisfy the
unconditional import-time checks.

1. Created a local "shared working dir" (used for cached Cartopy map assets):
   ```bash
   mkdir -p /home/sagemaker-user/weathergen_shared/hpc
   ```

2. Created a minimal private config file at
   `/home/sagemaker-user/.weathergen_private_conf.yml`:
   ```yaml
   path_shared_working_dir: /home/sagemaker-user/weathergen_shared
   ```

3. Created a stub `platform-env.py` at
   `/home/sagemaker-user/weathergen_shared/hpc/platform-env.py`:
   ```python
   def get_hpc() -> str | None:
       return None

   def get_hpc_user() -> str | None:
       return None

   def get_hpc_user_org() -> str | None:
       return None

   def get_hpc_config() -> str | None:
       return None

   def get_hpc_certificate() -> str | None:
       return None
   ```
   This satisfies the module-level `get_platform_env()` import. Its functions are
   only actually called if you pass `--push-metrics` to `evaluate` (MLflow
   upload), which this stub does not support.

4. Point the two env vars at these files before running `evaluate`:
   ```bash
   export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
   export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml
   ```
   (Not persisted to `~/.bashrc` — add them yourself if you want them set in every
   new shell.)

If you do have GitLab access to the real `WeatherGenerator-private` repo, the
better long-term fix is to clone it as a sibling of this repo
(`../WeatherGenerator-private`) instead of using this stub — see
`scripts/actions.sh` for how the default path is resolved, and
`packages/evaluate/README.md` for a link to the private wiki with fuller
workflow docs.

## 3. `Unexpected bus error encountered in worker` with `data_loading.num_workers > 0`

**Symptom:** running `inference` (e.g. via `scripts/inference_1deg.sh`) with
`data_loading.num_workers` set above `0` fails on the first batch with:
```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm)
```
This happens even with only a handful of workers and plenty of free GPU/system
memory, and is independent of worker count. It is **not** actual `/dev/shm`
exhaustion — on this box `/dev/shm` has ~3.95GB capacity with <1% used and no
cgroup memory limit. PyTorch emits this exact message for *any* abnormal
worker-process death signal, not just literal shm exhaustion, so it's
misleading here.

**Cause:** a classic "fork after starting threads" hazard between
`torch.multiprocessing`'s `"fork"` start method
(`torch.multiprocessing.set_start_method("fork", ...)` in
`src/weathergen/train/trainer_base.py:44`, via `Trainer.init_torch()`) and
`numcodecs`' blosc codec:

- `numcodecs/__init__.py` calls `blosc._init()` /
  `blosc.set_nthreads(min(8, cpu_count()))` **unconditionally at import
  time**, spinning up blosc's internal C pthread pool (up to 8 threads) the
  moment `weathergen.datasets` (→ `anemoi.datasets` → `zarr` → `numcodecs`)
  gets imported. This pool has no `os.register_at_fork` handler (unlike
  zarr's own `ThreadPoolExecutor`, which does register one in
  `zarr/core/sync.py`).
- `Trainer.inference()` constructs the dataset (`MultiStreamDataSampler` →
  `DataReaderAnemoi`, `src/weathergen/datasets/data_reader_anemoi.py:38-170`)
  in the main process, which does real zarr reads (dates, lat/lon,
  variables, normalization stats) that exercise blosc's multi-threaded pool
  — all before any fork.
- The actual fork happens later, when the `DataLoader` is first iterated
  (`iter(self.data_loader_validation)` in `Trainer.validate()`,
  `src/weathergen/train/trainer.py:581`). Forked worker processes inherit
  blosc's pthread/mutex bookkeeping but not the actual worker pthreads
  (`fork()` only clones the calling thread), so the first decompression call
  in a worker deadlocks on now-orphaned pthread state, surfacing as a bus
  error.

Versions in use: `zarr==3.1.6`, `numcodecs==0.16.5`.

**Fix (not yet applied — planned for later):** force blosc to run
single-threaded before the fork ever happens, so there's no multi-threaded
pool state to become invalid across `fork()`. Concretely: call
`numcodecs.blosc.set_nthreads(1)` in `TrainerBase.init_torch()`
(`src/weathergen/train/trainer_base.py`), right after
`torch.multiprocessing.set_start_method(...)`, guarded on
`multiprocessing_method == "fork"`. Once in place, `data_loading.num_workers`
in `scripts/inference_1deg.sh` / `scripts/inference_1deg_daily_loop.sh` can be
raised above `0` again (both currently hardcode `0` as a workaround for this
crash).

## 4. Score-map/GIF and per-init-hour timeseries silently come up empty when metric caches are out of sync

**Symptom:** running `evaluate` with `score_plots: [score_map, score_animation, timeseries, ...]`
(or the legacy `plot_score_maps`/`plot_score_animations`/`plot_score_init_timeseries` flags)
produces an empty `<run>/plots/<stream>/score_maps/` directory (no PNGs, no GIF) and no
`score_init_time_series/` directory at all — even though the regular summary plots (`lead_time`,
`heatmap`, `bar`, `scorecard`) render fine for the same metrics.

**Cause:** `_process_stream()` in
`packages/evaluate/src/weathergen/evaluate/run_evaluation.py:182-306` decides which metrics to pass
to `run_score_map_pipeline()`/`run_score_timeseries_pipeline()` based on
`reader.load_scores()` (`io/wegen_reader.py:192-239`), which returns `recomputable_metrics` — the
subset of requested metrics whose cached JSON score (`<metrics_dir>/<run_id>_<stream>_<region>_<metric>_chkpt*.json`)
is missing, stale (`eval_settings` mismatch), or doesn't have a matching `attrs`/parameter version
cached yet (`io/wegen_reader.py:241-291`).

If **any** requested metric needs recomputing while others are already fully cached,
`_process_stream` takes this branch:
```python
if recomputable_metrics:
    metrics_to_compute = recomputable_metrics   # only the stale/missing metric(s)
    regions_to_compute = list(set(recomputable_metrics.keys()))
elif plot_score_maps or plot_score_init_time_series:
    metrics_to_compute = {r: metrics for r in regions}   # the full set — only reached if NOTHING needs recomputing
    ...
```
So the score-map and per-init-timeseries pipelines get scoped to **only the metric(s) that needed
recomputing**, silently dropping every already-cached metric from those two outputs for that run —
even though the regular summary plots are unaffected (they read from the merged
`stream_loaded_scores` cache directly, not from this narrowed set).

We hit this because `psd` (in `evaluation.metrics`, used for spectral diagnostics) needed
recomputing (its cached `attrs` didn't match the current `psd_method` parameter) while `rmse`/`mae`
were still validly cached from an earlier run. That silently starved `rmse`/`mae` of score-map/GIF
and per-init-timeseries output on that run. On top of that, `psd` **also produces zero files by
itself** in the score-map pipeline — a power spectral density isn't a 2D lat/lon field, so it can't
render as a spatial map/GIF at all (the per-init-timeseries pipeline already knows this and
explicitly excludes it: `region_metrics.pop("psd", None)` in
`plotting/plot_orchestration.py:137`; the score-map pipeline has no equivalent guard).

**Fix:** don't mix `psd` (or any metric that can't produce a 2D score map) into the same
`evaluation.metrics` list as `score_map`/`score_animation`/`timeseries` outputs you actually want —
score it in a separate config/run instead. More generally: if score-map/GIF/per-init-timeseries
output for a metric comes up empty, check whether every metric in `evaluation.metrics` has a fully
warm, matching cache entry (compare the JSON files' mtimes under `<results>/evaluation/`) — a
partial cache miss on an unrelated metric will silently suppress these outputs for the metrics you
do care about.

## 5. `cartopy.io.DownloadWarning` crashes plotting for regional evaluation configs (e.g. `mozambique`)

**Symptom:** `evaluate` crashes while rendering scatter/map plots with:
```
cartopy.io.DownloadWarning: Downloading: https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_coastline.zip
```
raised inside a joblib worker (`plotter.py::scatter_plot` → `plt.savefig`) and re-raised
fatally by `dispatch_parallel`. Only happens for regions with a small bounding box (e.g.
`mozambique`, `belgium`, `arome`, `icon` — see
`packages/evaluate/src/weathergen/evaluate/utils/regions.py`); global/large-region plots
are unaffected.

**Cause:** Several things combine:

- `Plotter.__init__` (`packages/evaluate/src/weathergen/evaluate/plotting/plotter.py:145`)
  unconditionally calls `_download_cartopy_off(enabled=True)`, which does
  `warnings.filterwarnings("error", category=DownloadWarning)` (line 60) — turning any
  Cartopy download attempt into a hard error instead of an actual download. This is
  intentional (local-cache-only plotting), but there's no local fallback in place for
  every resolution.
- `ax.coastlines(linewidth=0.3)` (`plotter.py:1008`) uses the default `resolution='auto'`.
  Cartopy's auto-scaler picks shapefile resolution from map extent: small bounding boxes
  like `mozambique` (`regions.py:43`, `(-27, -10, 30, 41)`) resolve to `10m`, while
  large/global extents resolve to `50m`/`110m`.
- The Cartopy data dir is `<path_shared_working_dir>/assets/cartopy` (`plotter.py:46`),
  set up per fix #2 above — in this environment that's
  `/home/sagemaker-user/weathergen_shared/assets/cartopy`. The repo ships a pre-fetched
  cache at `WeatherGenerator/assets/cartopy/shapefiles/natural_earth/{physical,cultural}/`
  (the `.gitignore` comment there says "pre-fetched cartopy map assets (symlinked into
  the shared working dir)"), symlinked in via
  `weathergen_shared/assets/cartopy -> .../WeatherGenerator/assets/cartopy`. But that
  repo cache only ever contained `110m` shapefiles — no `50m`/`10m` — so any region small
  enough to trigger the auto-scaler's `10m` pick has nothing local to fall back to.
- The try/except around `ax.coastlines()` (`plotter.py:1007-1010`) doesn't help — Cartopy
  loads shapefiles lazily at draw time, so the actual failure happens later, unguarded,
  inside `plt.savefig` (`plotter.py:1095`).

Note this is **not** a network-egress problem — `naturalearth.s3.amazonaws.com` is
directly reachable from this SageMaker environment via plain `curl`/`wget`. It's
specifically the `_download_cartopy_off` warnings-as-errors policy that blocks Cartopy's
own downloader.

**Fix:** Pre-populate the missing resolution(s) directly with `curl`/`unzip` instead of
letting Cartopy try to download them itself. Only `coastline` is needed — nothing in the
`evaluate` plotting path calls `cfeature.LAND`/`OCEAN`/etc., just `ax.coastlines()`:
```bash
dir=/home/sagemaker-user/git/WeatherGenerator/assets/cartopy/shapefiles/natural_earth/physical
mkdir -p "$dir"
for res in 10m 50m; do
  curl -sL "https://naturalearth.s3.amazonaws.com/${res}_physical/ne_${res}_coastline.zip" -o /tmp/ne_${res}_coastline.zip
  unzip -o -q /tmp/ne_${res}_coastline.zip -d "$dir"
  rm /tmp/ne_${res}_coastline.zip
done
```
(Write into the repo's `assets/cartopy/` path, not directly into
`weathergen_shared/assets/cartopy/` — the latter is just a symlink to the former, per fix
above. If a fresh environment doesn't have that symlink yet, recreate it first:
`ln -s /home/sagemaker-user/git/WeatherGenerator/assets/cartopy /home/sagemaker-user/weathergen_shared/assets/cartopy`.)

If some other plot later starts using a different Cartopy feature (e.g. `LAND`, `OCEAN`,
`BORDERS`) and hits the same `DownloadWarning`, fetch that `name` at the needed
resolution(s) the same way — swap `coastline` for the feature's Natural Earth name (e.g.
`land`, `ocean`, `admin_0_countries`) and `physical` for `cultural` where applicable.

## After both fixes

`uv run evaluate --config config/evaluate/eval_test.yml` runs the full pipeline
and gets to real config errors (e.g. a missing `model_base_dir` for a given
`run_id`) rather than failing on environment setup. See
`docs/evaluate_config_reference.md` for what `eval_test.yml` needs per run_id.
