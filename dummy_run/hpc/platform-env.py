"""
Stub platform-env.py.

Normally this file lives in the separate WeatherGenerator-private repo and is
looked up via WEATHERGEN_PRIVATE_REPO_PATH (see
packages/common/src/weathergen/common/paths.py::get_wg_private_path). Both
`train`/`inference` and `evaluate` unconditionally check for this file's
directory to exist, even when data paths and private config are supplied
explicitly on the command line, so this stub exists purely to satisfy that
check for local, dataless experimentation. `get_hpc_config()` is only invoked
by `evaluate --push-metrics`, which this dummy workflow does not use.
"""


def get_hpc_config():
    raise NotImplementedError("No real platform-env.py configured; --push-metrics unsupported.")
