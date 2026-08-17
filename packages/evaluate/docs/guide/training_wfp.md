# Training Config Gotchas

## Starting a training run with `config/config_era5_o96_daily.yml`

```bash
cd /home/sagemaker-user/git/WeatherGenerator
export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml

uv run train --base-config=config/config_era5_o96_daily.yml --run-id <your-run-id>
```

- `--run-id` is optional but always worth setting explicitly — without it, a random id is
  generated for `results/`/`models/`/`logs/` and you have to dig it back out later (see
  `docs/inference_and_eval.md` §2).
- The two `WEATHERGEN_PRIVATE_*` env vars are this SageMaker environment's stub for the private
  HPC config that several modules load unconditionally at import time — see
  `docs/wfp_sagemaker_setup.md` §2 for what they point at and why they're needed.
- `--options key.path=value` overrides individual config values from the CLI without touching
  the file, e.g. `--options training_config.num_mini_epochs=2` for a short smoke run.

Notes from getting `config/config_era5_o96_daily.yml` ready for a real training run: a
duration-string format bug that silently breaks `time_window_step`/`time_window_len`, how
sample timing across `training_config`/`validation_config`/`test_config` actually works (and
why there's no way to get an exact equal-per-month sample count in a single run), and why you
can't set `batch_size` directly. See `docs/inference_and_eval.md` for the closely related
inference/evaluation sample-timing notes this document builds on.

---

## Resuming a killed or interrupted run

**`uv run train_continue`** resumes from a checkpoint instead of starting from scratch (`train`
and `train_continue` are separate entry points wired to `run_train`/`run_continue` in
`src/weathergen/run_train.py`, both dispatched via `cli.Stage` in `src/weathergen/utils/cli.py`).

```bash
cd /home/sagemaker-user/git/WeatherGenerator
export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml

uv run train_continue --from-run-id <old-run-id> --base-config=config/config_era5_o96_daily.yml \
  --run-id <old-run-id> --reuse-run-id
```

- **Find the run id first** if you don't already know it: checkpoints live under
  `$WEATHERGEN_PRIVATE_REPO_PATH/models/<run-id>/`, one pair of files per checkpointed
  mini-epoch (`<run-id>_chkptNNNNN.chkpt` + `model_<run-id>_chkptNNNNN.json`), plus a rolling
  `<run-id>_latest.chkpt`. `ls -lt` across `models/*/` and matching against the base-config name
  and last-modified time is the fastest way to identify which run id got killed.
- `--from-run-id` (required by `_add_model_loading_params`, `cli.py:140-148`) says which run's
  checkpoint to load from. `-e`/`--mini-epoch` picks which checkpoint of that run (default `-1` =
  latest, i.e. `<run-id>_latest.chkpt` — usually what you want, no need to pass it explicitly).
- `--run-id` sets where the *resumed* run's artifacts go. Pass `--reuse-run-id` to keep writing
  into the same `<old-run-id>` directories (simplest — same as the example above); omit it and
  give a fresh `--run-id` if you want the old run's artifacts left untouched and a new output
  directory started. Either way `--from-run-id` is what actually points at the checkpoint.
- Mini-epoch count picks up where the checkpoint left off, not from 0 — e.g. resuming a
  `chkpt00087` checkpoint continues toward the config's `num_mini_epochs` starting at 87, it
  doesn't add 87 more on top.
- `cf.general.run_history` records every `(from_run_id, istep)` this run was resumed from
  (`run_train.py:145`), so a chain of resumes stays traceable in the saved config.
- If the original run was launched through a wrapper script (e.g. `scripts/train_1deg.sh`, which
  also spawns `monitor_resources.py` alongside the training process), swap that script's `train`
  invocation for `train_continue` with the flags above rather than resuming the bare command by
  hand.

---

## 1. `D-HH:MM:SS` duration strings silently parse to a *negative* timedelta

**Symptom:** `TimeWindowHandler` raises `AssertionError: time window idxs invalid: 0 <= <negative
number>` during dataset init, or (less obviously) window arithmetic is just wrong.

**Cause:** every duration key (`time_window_step`, `time_window_len`, `forecast.time_step`) goes
through `parse_timedelta()` (`packages/common/src/weathergen/common/config.py:43-51`), which is a
thin wrapper around `pandas.to_timedelta`:

```python
def parse_timedelta(val: str | int | float | np.timedelta64) -> np.timedelta64:
    if isinstance(val, int | float | np.number):
        return np.timedelta64(pd.to_timedelta(val, unit="s")).astype("timedelta64[ms]")
    return np.timedelta64(pd.to_timedelta(val)).astype("timedelta64[ms]")
```

`pandas.to_timedelta` does **not** accept `"<days>-HH:MM:SS"` as a day-prefixed duration the way
you'd expect from, say, `datetime.timedelta` string conventions. It silently parses it as
something else entirely:

```python
>>> pd.to_timedelta("1-00:00:00")
Timedelta('-5 days +20:00:00')   # == -4.1666... days, not +1 day
```

Every other config in the repo already avoids this — they all use plain **total-hours
`HH:MM:SS`**, which pandas parses correctly even past 24 hours:

```python
>>> pd.to_timedelta("24:00:00")
Timedelta('1 days 00:00:00')
>>> pd.to_timedelta("384:00:00")   # 16 days
Timedelta('16 days 00:00:00')
```

`config/config_era5_o96_daily.yml` was the one exception (`time_window_step: 1-00:00:00`,
`time_window_len: 1-00:00:00`, `forecast.time_step: 1-00:00:00`) — likely a copy-paste from a
`datetime.timedelta`-style mental model rather than this codebase's convention. With that value,
`TimeWindowHandler.get_index_range()` (`src/weathergen/datasets/data_reader_base.py:108-126`)
computes a negative `idx_end`, and the very next line's assertion fails immediately — the config
could not start a training run at all until fixed.

**Fix:** always write duration config values as total-hours `HH:MM:SS`, never `D-HH:MM:SS`, e.g.
`"24:00:00"` for 1 day, `"365:00:00"` for ~15.2 days. If a duration ever needs to exceed 24
hours, just keep adding to the hours field — `HH` isn't clamped to 24.

---

## 2. Sample timing: no month-aware sampler exists

**Background:** `start_date`/`end_date`/`time_window_step` define a linear sequence of valid
time-window indices (`TimeWindowHandler`, `data_reader_base.py:67-145` — index `i` and `i+1` are
always exactly `time_window_step` apart in wall-clock time, with no concept of calendar months).
`MultiStreamDataSampler` (`src/weathergen/datasets/multi_stream_data_sampler.py`) then draws
`samples_per_mini_epoch` samples from that index range one of two ways, controlled by `shuffle`:

- **`shuffle: True`** (training default): every mini-epoch, `reset()` takes a **uniform random
  permutation of every valid index in the full range** (`multi_stream_data_sampler.py:282-309`),
  then uses the first `samples_per_mini_epoch` of that shuffled order. Over many mini-epochs this
  averages out to roughly uniform coverage of the year without any further tuning — this is why
  `training_config` here is left alone (`samples_per_mini_epoch: 72`, `shuffle: True`, 100
  mini-epochs).
- **`shuffle: False`** (validation/test default): samples are drawn **strictly sequentially**
  from `start_date`, spaced by `time_window_step`, for however many indices fit
  `samples_per_mini_epoch` (`multi_stream_data_sampler.py:786`, confirmed in
  `docs/inference_and_eval.md` §5). There is **no month-aware or stratified sampling anywhere in
  this codebase** — confirmed by grepping `src/`/`packages/` for "month"/"stratif" (only
  unrelated MLflow dashboard code turns up). Concretely, this means with the default inherited
  1-day `time_window_step` and `samples_per_mini_epoch: 24`, validation only ever sampled **the
  first 24 days of January**, every single epoch — nothing later in the year was ever validated
  against.

**`validation_config` runs automatically every mini-epoch** from one fixed date range/stride
(`Trainer.train()` calls `self.validate(mini_epoch, self.validation_cfg, ...)` each iteration,
`src/weathergen/train/trainer.py:402`) — there's no way to split it into per-month sub-runs
within one training job; the whole run shares one `TimeWindowHandler`.

**`test_config` is different**: it's only read by the standalone `uv run inference`/`test` entry
point (`Trainer.inference()` → `self.validate(0, self.test_cfg, ...)`, `trainer.py:247`), which
is a separate CLI invocation each time. So if you ever need an **exact** equal count per
calendar month (not just "spread out"), that's the only way to actually get it: run `test_config`
12 times, once per month, scoping each with CLI overrides —

```bash
--options test_config.start_date=2023-03-01T00:00 test_config.end_date=2023-03-31T00:00 \
          test_config.samples_per_mini_epoch=2
```

— then aggregate/average the resulting scores afterward. This is not possible for the in-loop
`validation_config`.

**What this config actually does** (a deliberate approximation, not exact-per-month): both
`validation_config` and `test_config` (which inherits `validation_config`'s settings via
`test_cfg = merge(merge(training_config, validation_config), test_config)`,
`trainer.py:126-131`) now set

```yaml
time_window_step: 365:00:00   # ~15.2 days
```

With `samples_per_mini_epoch: 24` and `shuffle: False`, this spaces the 24 sequential samples
roughly twice a month across the full year, instead of front-loading them all into the first
~24 days of the range. Checked concretely for `validation_config`'s 2022 range: samples land in
**every** calendar month, with counts `{Jan: 3, Feb: 1, Mar–Dec: 2 each}` — a fixed stride still
drifts relative to real month boundaries (months aren't all the same length), so it's not an
exact 2-per-month split, but it's a large improvement over the original January-only behavior at
zero extra validation cost (still 24 samples/epoch, not ~365).

---

## 3. Batch size can't be set directly

**There is no `batch_size` config key anywhere in this codebase.** Effective per-GPU batch size
is *derived* by summing `num_samples` across every enabled entry under a stage's `model_input`
block:

```python
# src/weathergen/train/utils.py:139-150
def get_batch_size_from_config(config: Config) -> int:
    num_samples = 0
    for _, source_cfg in config.model_input.items():
        if source_cfg.get("enabled", True):
            num_samples += source_cfg.get("num_samples", 1)
    assert num_samples > 0, "Number of samples in source configs needs to greater than 0."
    return num_samples
```

- `num_samples` defaults to `1` if omitted from a `model_input` entry.
- This config sets `model_input.forecasting.num_samples: 2` → per-GPU batch size is `2`.
- `Trainer.init()` computes this separately per stage: `self.batch_size_per_gpu`,
  `self.batch_size_validation_per_gpu`, `self.batch_size_test_per_gpu` (`trainer.py:134-137`).
- **Effective global batch size** is `world_size * batch_size_per_gpu`
  (`Trainer.get_batch_size_total()`, `trainer.py:94-98`), and it's not just a data-loading
  number — it feeds directly into AdamW `beta1`/`beta2`/`eps` scaling and the LR scheduler
  (`trainer.py:323-356`).

**The DataLoader itself does no batching.** Both the training and validation/test loaders are
built with:

```python
loader_params = {"batch_size": None, "batch_sampler": None, "shuffle": False, ...}
torch.utils.data.DataLoader(self.dataset, **loader_params, sampler=None)
```

`batch_size=None`/`batch_sampler=None` disables PyTorch's own batching — `MultiStreamDataSampler`
is an `IterableDataset` that already yields fully-assembled batches from `__iter__`
(`multi_stream_data_sampler.py:778-803`, `self.batch_size` samples at a time). **The only lever
is `model_input.<source>.num_samples` in the relevant stage config** — there is no
`DataLoader(..., batch_size=N)` to reach for.

**The gotcha:** `check_samples()` (`multi_stream_data_sampler.py:156-210`) requires
`samples_per_mini_epoch >= world_size * batch_size_per_gpu`. If it's smaller, it's silently
bumped up with a warning:

```
samples_per_mini_epoch=<N> is too small for world_size=<W> and batch_size=<B>. ...
Automatically increasing to <W*B>.
```

If, after that adjustment, `repeat_data_in_mini_epoch: False` (the default) and the date range
still can't supply that many samples without duplicating some per rank, it hits a **hard,
uncorrected** assertion right below (`assert n_duplicates <= 0`, line 210) and the run crashes.
Widening the date range, lowering `samples_per_mini_epoch` (down to the `world_size*batch_size`
floor), or setting `repeat_data_in_mini_epoch: True` are the ways out.

**Unrelated look-alike:** there's a *separate* `assert batch_size == 1` in
`src/weathergen/model/parametrised_prob_dist.py:111-115` (`interpolate_with_noise`). That's the
per-cell local-assimilation chunk size inside the latent-noise/diffusion-style decoder path
(`latent_noise_*` config options), not the training batch size described above — only relevant
if `latent_noise_gamma > 0` (this config keeps it at `0.0`).
