#!/usr/bin/env python3
"""Sample GPU/CPU/RAM usage at a fixed interval while running a wrapped command.

Usage:
    uv run --directory /home/sagemaker-user/git/WeatherGenerator python scripts/monitor_resources.py \\
        --interval 2 --output results/<run_id>/resource_usage_<stage>.csv \\
        -- uv run --directory /home/sagemaker-user/git/WeatherGenerator inference --base-config=... [...]

Exits with the wrapped command's exit code.
"""

import argparse
import csv
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

CSV_HEADER = [
    "timestamp",
    "cpu_percent",
    "ram_used_gb",
    "ram_total_gb",
    "gpu_util_percent",
    "gpu_mem_used_gb",
    "gpu_mem_total_gb",
]


def sample_gpu() -> tuple[float, float, float] | None:
    """Query the first GPU's utilization/memory via nvidia-smi, or None if unavailable."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    gpu_util, mem_used, mem_total = (float(x) for x in out.splitlines()[0].split(","))
    return gpu_util, mem_used / 1024, mem_total / 1024


def sample_row() -> list:
    cpu_percent = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    row = [
        time.strftime("%Y-%m-%dT%H:%M:%S"),
        cpu_percent,
        round(vm.used / 1e9, 2),
        round(vm.total / 1e9, 2),
    ]
    gpu = sample_gpu()
    row += list(gpu) if gpu else ["", "", ""]
    return row


def monitor_loop(interval: float, output_path: Path, stop_event: threading.Event) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        # prime psutil.cpu_percent (first call after init always returns 0.0)
        psutil.cpu_percent(interval=None)
        while not stop_event.wait(interval):
            writer.writerow(sample_row())
            f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--interval", type=float, default=2.0, help="Sampling interval in seconds (default: 2).")
    parser.add_argument("--output", required=True, type=Path, help="CSV path to write resource samples to.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to run, prefixed with `--`.")
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given; pass it after `--`")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_loop, args=(args.interval, args.output, stop_event), daemon=True
    )
    monitor_thread.start()

    start = time.time()
    result = subprocess.run(command)

    stop_event.set()
    monitor_thread.join(timeout=args.interval + 5)

    elapsed = time.time() - start
    print(f"[monitor_resources] wrote resource log to {args.output} ({elapsed:.1f}s run)")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
