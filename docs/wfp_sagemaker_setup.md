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

## After both fixes

`uv run evaluate --config config/evaluate/eval_test.yml` runs the full pipeline
and gets to real config errors (e.g. a missing `model_base_dir` for a given
`run_id`) rather than failing on environment setup. See
`docs/evaluate_config_reference.md` for what `eval_test.yml` needs per run_id.
