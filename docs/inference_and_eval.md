# Inference & Evaluation: How It Actually Works

Notes from tracing `uv run inference` and `uv run evaluate` through the code, prompted by
confusion over how many forecast runs (samples) and how many lead-time steps a given
`scripts/inference_1deg.sh` invocation actually produces, and where to verify that in the
output. See `docs/evaluate_config_reference.md` for the full `evaluate` config schema — this
document is about the `inference` side and how the two connect.

---

## 1. `inference` always runs the validation code path

`inference` (`pyproject.toml` → `weathergen.run_train:inference`) loads model weights from
`--from-run-id <checkpoint run>`, builds a merged config, and calls
`Trainer.inference()` (`src/weathergen/train/trainer.py:193-248`), which internally just calls
`self.validate(0, ...)`. There is no separate "inference-only" data path — it reuses the same
validation/scoring machinery, always writing `mini_epoch=0` in the output filename regardless
of which checkpoint mini-epoch was actually loaded via `--mini-epoch`.

## 2. Run ID: the output folder name is *not* `--from-run-id`

- `--from-run-id 8level_1deg_2020_smoke2` tells `inference` which trained checkpoint to load.
- Unless you pass `--run-id <name>` (or `--reuse-run-id`), a **brand-new random id** is
  generated for this run's own output (e.g. `ui1udtjn`, via `get_run_id()` in
  `packages/common/src/weathergen/common/config.py:169`) and used for
  `results/<new_run_id>/`, `models/<new_run_id>/`, and `logs/<new_run_id>/`.
- The mapping back to the source checkpoint is recorded in
  `models/<new_run_id>/model_<new_run_id>_chkpt00000.json` under the `from_run_id` field and
  `general.run_history`. To find which generated id came from a given `--from-run-id`:
  ```bash
  grep -l '"from_run_id": "8level_1deg_2020_smoke2"' /home/sagemaker-user/weathergen_shared/models/*/model_*.json
  ```
- **Recommendation:** always pass an explicit `--run-id <descriptive-name>` for anything you'll
  want to find again later — see `scripts/inference_1deg.sh`, which uses
  `--run-id era5_1deg_daily96h`.

## 3. What controls the number of forecast runs (samples)

Three config sections stack, each overriding the previous for the corresponding stage
(`trainer.py:118-132`):

```
test_cfg = merge(merge(training_config, validation_config), test_config)
```

The **number of forecast runs per `inference` invocation is `test_cfg.samples_per_mini_epoch`**.
In `config/config_era5_2020_smoke.yml`, `test_config` does not set this key, so it is inherited
from `validation_config.samples_per_mini_epoch`. `test_config.output.num_samples` looks like it
should control this but it does **not** — it only gates how many of the *already-produced*
batches get written to disk (`trainer.py:583`, `627-645`); if it's larger than the actual number
of samples produced, it has no visible effect at all.

To control the count without touching the shared base config (which is also used for real
training runs), override it via CLI, scoped to `test_config` only:

```bash
--options test_config.samples_per_mini_epoch=60
```

You don't need to compute the exact number that fits your date range — `check_samples()`
(`src/weathergen/datasets/multi_stream_data_sampler.py:158-190`) automatically clamps an
over-large request down to whatever actually fits between `start_date`/`end_date` and the
forecast horizon, and logs a warning with the real count it used. Check the run's log
(`weathergen_shared/logs/<run_id>/`) for that line to see exactly how many samples were run.

## 4. What controls the number of forecast timesteps (lead-time rollout)

Same inheritance chain applies to `forecast`: **`test_cfg.forecast.num_steps`**, inherited from
`training_config.forecast.num_steps` unless overridden. Each step advances by
`forecast.time_step` (e.g. `06:00:00`). The actual number of output steps written is
`offset + num_steps` (`multi_stream_data_sampler.py:632-634`), where index `0` is always the
source/init condition — so `offset: 1, num_steps: 1` (the smoke-test default) produces exactly
two entries per sample: `ERA5/0` (source) and `ERA5/1` (the one 6h-lead prediction).

To extend the horizon, override for `test_config` only:

```bash
--options test_config.forecast.num_steps=16   # 16 * 6h time_step = 96h horizon
```

This merges onto the inherited `forecast` block (keeping `time_step`, `policy`, `offset` as
configured for training) — `training_config` itself, and therefore actual training runs, are
unaffected.

**Caveat:** a model trained with `forecast.num_steps=1` (as `8level_1deg_2020_smoke2` was) only
ever saw single-step targets during training. Running a longer autoregressive rollout at
inference is exactly how you observe error growth with lead time, but don't be surprised if
skill degrades faster than a model actually trained for multi-step rollout — that's the
expected signal, not a bug.

## 5. Sample timing is deterministic, not random — and how to land on 00Z daily

`validation_config.shuffle: False` and `test_config` doesn't override it, so `test_cfg.shuffle`
is `False`. In `multi_stream_data_sampler.py:308-309`, permutation shuffling (`rng.permutation`)
only happens when `shuffle=True` — otherwise samples are drawn **sequentially in time order**
starting at `test_config.start_date`, spaced by `test_cfg.time_window_step` (inherited from
`training_config.time_window_step: 06:00:00` unless overridden). That's why the original 8
samples landed on every synoptic hour (00/06/12/18Z) across 2 days rather than once/day.

To get one init per day at 00Z (given `start_date` is already `...T00:00`), override the stride
for `test_config` only:

```bash
--options test_config.time_window_step=24:00:00
```

Since iteration walks `perms` sequentially from the front (`multi_stream_data_sampler.py:786`)
and the count is bounded by the correctly-computed `check_samples()` margin, this is safe even
though a separate internal helper (`_calc_baseperms`) computes its own margin with an integer
division that would truncate to zero when `time_window_step > forecast.time_step` — that
helper's output is never actually indexed into far enough to hit the truncated region in
practice, because `self.len` (from `check_samples()`) is always smaller.

## 6. `config/config_era5_2020_smoke.yml`'s ERA5 data range

The smoke-test data (`/home/sagemaker-user/data/era5/era5_1deg_2020.zarr`) covers exactly
`2020-01-01T00:00` through `2020-12-31T18:00` (1464 six-hourly timesteps) — there is no slack to
pad `end_date` further past the smoke config's existing `test_config.end_date: 2020-12-31T18:00`.

## 7. Verifying what actually ran, from the output files

Don't infer counts from console logs alone — the output itself is authoritative:

**Zarr-in-zip output** (`results/<run_id>/validation_chkpt00000_rank0000.zip`):
```bash
# number of samples (top-level dirs, one per init time)
unzip -l results/<run_id>/validation_chkpt00000_rank0000.zip | grep -c '^[0-9]*/zarr.json$'

# number of steps within one sample (0 = source, 1..N = forecast steps)
unzip -l results/<run_id>/validation_chkpt00000_rank0000.zip | grep -E '^0/ERA5/[0-9]+/zarr.json$'
```
It's Zarr data stored inside a zip (not just a plain archive) — open it directly with
`zarr.open("results/<run_id>/validation_chkpt00000_rank0000.zip", mode="r")` or
`xarray.open_zarr(...)` rather than fully extracting it.

**Evaluation score JSON** (`results/<run_id>/evaluation/<run_id>_<stream>_<region>_<metric>_chkpt00000.json`)
is the most direct source of truth — it records exactly what was scored:
```python
import json
d = json.load(open("results/<run_id>/evaluation/<run_id>_ERA5_global_rmse_chkpt00000.json"))
d["eval_settings"]["forecast_steps"]        # e.g. [1, 2, ..., 16]
d["scores"][0]["coords"]["sample"]["data"]  # sample indices actually scored
d["scores"][0]["coords"]["lead_time"]["data"]   # lead time in hours per forecast_step
d["scores"][0]["coords"]["init_times"]["data"]  # actual init datetimes — confirms 00Z daily cadence
d["scores"][0]["coords"]["ens"]["data"]     # ensemble members present (this codebase: always [0])
```

**Per-run log** (`weathergen_shared/logs/<run_id>/`): look for the `TimeWindowHandler` info line
(confirms date range/stride actually used) and any `check_samples()` warning (confirms the
actual, possibly-clamped, sample count).

---

## 8. Data-loading performance: why it can bottleneck, and what's tunable

The `inference` DataLoader is built in `Trainer.inference()`
(`src/weathergen/train/trainer.py:204-223`):

```python
self.dataset = MultiStreamDataSampler(cf, self.test_cfg, stage=VAL)
loader_num_workers = min(self.test_cfg.samples_per_mini_epoch, cf.data_loading.num_workers)
loader_params = {
    "batch_size": None,
    "batch_sampler": None,
    "shuffle": False,
    "num_workers": loader_num_workers,
    "pin_memory": cf.data_loading.get("memory_pinning", False),
    "persistent_workers": cf.data_loading.get("persistent_workers", False),
}
self.data_loader_validation = torch.utils.data.DataLoader(self.dataset, **loader_params, sampler=None)
```

Batching happens *inside* `MultiStreamDataSampler.__iter__` (it's an `IterableDataset` yielding
already-assembled `ModelBatch` objects), so there's no separate collation cost — the two knobs
that actually matter for throughput are `num_workers` and `pin_memory`, both sourced from the
top-level `data_loading:` config block (not stage-specific, unlike `training_config`/
`validation_config`/`test_config`).

**Why `num_workers: 0` (the smoke config's default) is slow:** with zero workers, PyTorch runs
the dataset's `__iter__`/`_get_batch` directly in the main process, on demand, each time
`validate()`'s loop asks for the next batch (`trainer.py:589`). Every zarr read + zstd
decompression + tokenization + masking happens synchronously, with **no overlap with GPU
compute** — the GPU sits idle while the CPU decompresses the next sample, then the CPU sits idle
while the GPU runs the forward pass. `MultiStreamDataSampler.worker_workset()`
(`src/weathergen/datasets/multi_stream_data_sampler.py:808-841`) already implements correct
worker-range partitioning via `torch.utils.data.get_worker_info()`, so raising `num_workers` is
a safe, already-supported flip — no code changes needed. Most other configs in this repo (e.g.
`config/default_config.yml`) already run with `num_workers: 12`; only the smoke config uses `0`.

**Why the 96h/16-step horizon made this more noticeable:** `collect_datasources`
(`multi_stream_data_sampler.py:52-83`) issues one `get_target` read per forecast step, landing in
`DataReaderAnemoi._get()` (`src/weathergen/datasets/data_reader_anemoi.py:216`), which reads and
decompresses one full zarr chunk per call — confirmed chunk shape `[1, 52, 1, 65160]` (time,
variable, ensemble, cell), i.e. one chunk = one 6h timestep across **all** 52 variables and all
65160 cells (~13.5MB decompressed), decompressed unconditionally even though only 2 channels
(`2t`, `10u`) are used downstream — channel selection happens in-memory *after* decompression.
With `forecast.num_steps=1` that's 2 reads/sample (source + 1 target); with `num_steps=16` it's
~17 reads/sample. The zarr chunking itself is well-aligned to the 6h access pattern (each read
touches exactly one chunk, no wasted adjacent-chunk fetches), so this isn't a chunking problem —
it's that 16x more synchronous, serialized decompression work now happens per sample under
`num_workers=0`.

**What's set in `scripts/inference_1deg.sh`:**
```bash
--options \
  data_loading.num_workers=2 \
  data_loading.memory_pinning=true
```
- `num_workers=2`: this SageMaker Studio box has only 4 vCPUs, so keeping this well under the
  core count leaves headroom for the main process and other work; going too high would
  oversubscribe the box. Adjust to your actual vCPU count elsewhere (`nproc`) minus 1-2.
- `memory_pinning=true` → `pin_memory=True` on the DataLoader, speeding up host→device transfer
  overlap. Safe here since this is a pure inference/`validate()` path; a documented caveat
  elsewhere in the repo (`config/default_config.yml`) about pinned memory causing hangs is
  specific to FSDP2+DINOv2 training and doesn't apply to inference.

**Explicitly out of scope for now (documented as backlog, not implemented):**
- `prefetch_factor` isn't exposed anywhere in the codebase (would need a small change to
  `loader_params` in `trainer.py:213-220` plus a new `data_loading.prefetch_factor` config key).
- `anemoi-datasets` supports an `LRUStoreCache` via `open_zarr(path, cache=...)`
  (`.venv/.../anemoi/datasets/data/stores.py:76-96`), but `DataReaderAnemoi.__init__`
  (`data_reader_anemoi.py:74,93`) never passes a `cache` argument, so it's unreachable today.
  Wiring it through could help if the same chunks get revisited across mini-epochs/streams, but
  on this 15GB-RAM box the ~12GB dataset already mostly lives in the OS page cache, so the payoff
  is uncertain relative to the code change involved.
- `persistent_workers` is read (`trainer.py:219`) but never set in any config, so it's
  effectively dead — it only matters when a loader is iterated more than once per process
  lifetime, which doesn't apply to a single one-shot `inference()` call.

---

## 9. Can a stream read data straight from S3?

Short answer: **not today, and not by mounting the bucket either — in this specific
environment.** If you're handed an S3 bucket URL for a dataset, here's what actually works.

### Native `s3://` paths in `filenames:` don't work

`MultiStreamDataSampler._init_stream_datasets` (`src/weathergen/datasets/multi_stream_data_sampler.py:245-260`)
wraps every `filenames:` entry in `pathlib.Path(...)` and checks `.exists()` *before*
`anemoi_datasets.open_dataset()` is ever reached:

```python
for fname in stream_info.get("filenames", [pathlib.Path()]):
    fname = pathlib.Path(fname)
    if fname.exists():
        filename = fname
    else:
        filenames = [pathlib.Path(path) / fname for path in cf.data_paths]
        filename = next((f for f in filenames if f.exists()), None)
        if filename is None:
            raise FileNotFoundError(...)
```

`pathlib.Path("s3://bucket/key.zarr").exists()` does a local `stat()` and returns `False` — this
raises `FileNotFoundError` immediately, before any S3-aware code path gets a chance to run.

Even bypassing that (there's an alternate `anemoi_config` stream key that skips the `filenames`/
`.exists()` path entirely and passes straight through to `anemoi_datasets.open_dataset()` — see
`data_reader_anemoi.py:60-74` — but no existing config in this repo uses it), anemoi-datasets'
zarr3 `S3Store` (`.venv/.../anemoi/datasets/compat/zarr3.py`) needs the `obstore` package, which
is **not installed** — `pyproject.toml`/`uv.lock` declare no S3-related dependency at all (no
`obstore`, `boto3`, `s3fs`). So two independent things would need to change: the local-path
check in the sampler, and a new dependency.

### FUSE mounting doesn't work in this SageMaker Studio environment either

The obvious workaround — mount the bucket as a local path via `mountpoint-s3`, `s3fs-fuse`, or
`goofys`, then point `filenames:` at the mount point exactly like any other local path — is
blocked here specifically: this SageMaker Studio app runs in a container without FUSE device
access. Verified directly on this box:
```bash
$ ls /dev/fuse           # No such file or directory
$ sudo modprobe fuse     # modprobe: command not found
$ cat /.dockerenv        # present — confirms containerized
```
`/dev/fuse` isn't present and there's no way to add it from inside the container (it has to be
granted at container-launch time via `--device /dev/fuse` or similar, which isn't under user
control here). This rules out all FUSE-based S3 filesystem tools, not just one of them.

### What actually works today: sync a local copy

The SageMaker execution role already has working AWS credentials (`aws sts get-caller-identity`
succeeds, `aws-cli/2.35.11` is installed), and `/home/sagemaker-user` had ~59GB free at time of
writing. The simplest path that requires **zero code changes**:

```bash
aws s3 sync s3://<bucket>/<prefix>/dataset.zarr /home/sagemaker-user/data/<name>.zarr
```

Then point a stream yaml's `filenames:` at the local copy — identical to the existing pattern in
`config/streams/era5_2020_smoke/era5.yml:10` (`filenames: ['/home/sagemaker-user/data/era5/era5_1deg_2020.zarr']`).
Check the bucket's size against free disk space first (`aws s3 ls --summarize --human-readable
--recursive s3://<bucket>/<prefix>/` and `df -h /home/sagemaker-user`) — this only makes sense if
the dataset comfortably fits locally and doesn't change so often that re-syncing becomes a
burden.

### If the dataset is too large to copy, or changes frequently

Native `s3://` support is possible but bigger in scope, and **unimplemented/untested** in this
repo: add `obstore` as a dependency, and patch the `.exists()` check in
`multi_stream_data_sampler.py:245-260` to bypass the local-filesystem check for `s3://` (and other
URL-scheme) paths so they pass through to `anemoi_datasets.open_dataset()` unchanged. This would
need real testing of credential plumbing (`anemoi.utils.remote.s3.s3_options`) and read
performance (repeated network reads per sample vs. local NVMe) before relying on it.

---

## 10. Resource usage logging (GPU/CPU/memory)

Nothing in the training/evaluation code logs GPU/CPU/RAM usage — `mlflow` here only tracks
scores/loss, and its built-in automatic system-metrics logging is never enabled. Rather than
wire that into `trainer.py`/`run_evaluation.py`, `scripts/monitor_resources.py` is a standalone
wrapper: it launches the actual `uv run ... train|inference|evaluate ...` command as a subprocess
and, on a background thread, samples system-wide CPU% and RAM (`psutil`, already a project
dependency) and GPU utilization/memory (`nvidia-smi`, present on this box) every `--interval`
seconds (default 2s) until the wrapped command exits, writing one row per sample to a CSV. It
exits with the wrapped command's exit code, so it's transparent to scripting/CI. If `nvidia-smi`
isn't found (e.g. a non-GPU box), the GPU columns are just left blank rather than failing.

`scripts/train_1deg.sh`, `scripts/inference_1deg.sh`, and `scripts/evaluate_1deg.sh` all wrap
their run through it, e.g.:

```bash
RUN_ID=era5_1deg_daily96h

uv run --directory /home/sagemaker-user/git/WeatherGenerator python scripts/monitor_resources.py \
  --interval 2 \
  --output "$WEATHERGEN_PRIVATE_REPO_PATH/results/${RUN_ID}/resource_usage_inference.csv" \
  -- \
  uv run --directory /home/sagemaker-user/git/WeatherGenerator inference ...
```

The CSV lands inside that run's own `results/<run_id>/` folder (`resource_usage_train.csv` /
`_inference.csv` / `_evaluate.csv`), next to the validation zip / evaluation JSONs / plots for
that same run — see section 7 for the rest of what lives there. Re-running a script overwrites
its own resource CSV, same as re-running overwrites that run's other output (since `--run-id` is
fixed per script). `evaluate_1deg.sh` doesn't pass a `--run-id` CLI flag (it's config-driven), so
its `RUN_ID` shell variable exists only to name the log path and must be kept in sync by hand with
whichever `run_ids:` entry `config/evaluate/eval_test.yml` is actually scoring.

To inspect a log:
```python
import pandas as pd
df = pd.read_csv("results/era5_1deg_daily96h/resource_usage_inference.csv", parse_dates=["timestamp"])
df.plot(x="timestamp", y=["cpu_percent", "gpu_util_percent"])
```
Or watch it live while a run is in progress: `tail -f results/<run_id>/resource_usage_*.csv` (rows
are flushed after every sample, not just at the end).

---

## 11. Benchmarking against IFS/AIFS forecasts as pseudo run_ids

Investigated after wanting `config/evaluate/eval_test.yml` to show `era5_1deg_daily96h`
alongside ECMWF's own operational IFS/AIFS forecasts for the same period, without building a
full pipeline by hand.

### What actually works today, with zero repo code changes

The eval framework's `type:` field on a `run_ids` entry already supports plugging in a model
that isn't a WeatherGenerator run at all, as long as its scores are pre-computed and written to
CSV in the Quaver column convention `CsvReader` expects (`packages/evaluate/src/weathergen/
evaluate/io/csv_reader.py`, format documented in `docs/evaluate_config_reference.md` §11):
`parameter,level,number,score,step,date,domain_name,value`, one file per metric under
`<metrics_dir>/<run_id>/`. `config/evaluate/eval_config.yml:225-249` already has live examples
of exactly this for `pangu` (Pangu-Weather) and `graphcast`.

So the practical path is: pull IFS/AIFS forecasts and score them against ERA5 truth in a
**standalone script, outside `evaluate` entirely** (e.g. via `earthkit-data`'s
`"ecmwf-open-data"` source — already a project dependency, see `packages/evaluate/pyproject.
toml`), write the results as CSVs in the format above, then add a `type: "csv"` run_ids entry
pointing at them — identical to `pangu`/`graphcast`. No new Reader class, no dependency changes,
no dispatch changes in `run_evaluation.py`.

**Caveat found while checking this**: `CsvReader.__init__` (`csv_reader.py:73-80`) always builds
the channel key as `<parameter>_<level>` and drops any row with a null `level`
(`dropna(subset=["step", "level"])`) — there's no code path that produces a bare channel name
without a level suffix. This works cleanly for pressure-level channels (`z_500`, `t_850`, ...,
matching WeatherGenerator's own naming), but **surface channels (`2t`, `10u`) can't be
represented as-is** — whatever `level` value you write, the resulting channel key (e.g. `2t_0`)
won't match the plain `"2t"` in the eval config's `channels:` list, so those rows are silently
dropped and the channel shows up as missing/NaN rather than erroring. If a surface-channel
comparison matters, either restrict the benchmark run_ids to pressure-level channels only, or
special-case `level is null → channel = parameter` in `CsvReader` (small, self-contained change).

Also note: like the other non-`"zarr"` reader types, `csv` is score-only — `score_map`/
`score_animation` (spatial maps/animations) won't be produced for these pseudo run_ids, only the
score-vs-lead-time/timeseries/heatmap/scorecard/bar plots. This matches what's wanted for a
benchmark comparison.

### Backlog: live-pulling reader (`type: "ecmwf_opendata"`)

A fancier option was designed but **not implemented** — a first-class reader type that pulls
IFS/AIFS fields from ECMWF's open-data feed and scores them on the fly inside `evaluate` itself
(rather than via an offline script + CSV), so re-running `evaluate` automatically picks up new
init times without a separate manual step. Sketch, if ever revisited:

- New reader `EcmwfOpenDataReader(Reader)`, registered as `type: "ecmwf_opendata"` in
  `get_reader()` (`run_evaluation.py:158-180`), score-only like `CsvReader` (no maps/animations).
- A `target_run_id` config key names an existing `type: "zarr"` run_id (e.g.
  `era5_1deg_daily96h`) to borrow ERA5 truth + grid + forecast_steps/samples from — necessary
  because `WeatherGenZarrReader` only reads truth pre-baked into a real inference run's own
  results zarr (`io_orchestration.py`/`io_workers.py:139-268`), there's no standalone in-repo way
  to open raw ERA5 by (channel, datetime). Internally instantiate a second
  `WeatherGenZarrReader(self.eval_cfg, target_run_id, self.private_paths)`, same pattern
  `WeatherGenMergeReader` already uses (`io/merge_reader.py:56-79`).
- Pull via `earthkit.data.from_source("ecmwf-open-data", model="ifs"|"aifs", ...)` (needs the
  separate `ecmwf-opendata` PyPI client added as a dependency — not installed today), map
  WeatherGenerator channel names to ECMWF request params via the existing
  `config/evaluate/config_zarr2cf.yaml` table, regrid onto the target run's `ipoint` lat/lon
  with `scipy.interpolate.RegularGridInterpolator` (already a direct dependency, no new
  regridding dep needed), score with the framework's own `get_score()`
  (`evaluate/scores/score.py:123-169`) instead of reimplementing rmse/mae.
- Two-layer disk cache: raw pulled+regridded fields (keyed by run_id/init_time/channel/step, so
  repeat `evaluate` runs don't re-hit ECMWF's server) and the usual score JSON cache
  (`<metrics_dir>/<run_id>_<stream>_<region>_<metric>_chkpt00000.json`, same convention as the
  other reader types).
- **Hard constraint regardless of implementation**: ECMWF's free open-data feed only retains a
  rolling ~4-day window, so this only ever works against *recent* inference runs, never the
  original 2020 smoke-test period — `era5_1deg_daily96h` (or whichever run_id is used as
  `target_run_id`) would need to be re-run against current dates for a live comparison to have
  anything to align against.

Full design detail (regridding/caching specifics, exact file layout) was written up but not
committed to code; revisit if the CSV-based route above turns out to be too manual in practice.

---

## Current setup in this repo

- `scripts/inference_1deg.sh`: loads `8level_1deg_2020_smoke2`, writes output to
  `results/era5_1deg_daily96h/`, one init per day at 00Z (`test_config.time_window_step=24:00:00`),
  96h horizon (`test_config.forecast.num_steps=16`), up to 60 samples (auto-clamped to whatever
  fits Dec 2020), with `data_loading.num_workers=2` and `data_loading.memory_pinning=true` to
  overlap data loading with GPU compute (see section 8).
- `config/evaluate/eval_test.yml`: scores `run_ids.era5_1deg_daily96h` with `forecast_step`/
  `sample`/`ensemble` all `"all"` (exhaustive scoring), while `plotting` is restricted to a
  representative subset (`forecast_step: [1,4,8,12,16]`, `sample: [0,10,20]`) to keep plot volume
  manageable. Verbosity is turned up: `print_summary`, `plot_score_maps`,
  `plot_score_animations`, `heat_maps`, `score_cards`, and `bar_plots` are all enabled, alongside
  the pre-existing `summary_plots` and `plot_score_init_timeseries` (the score-vs-lead-time line
  plots that show error growth over the 96h horizon).
