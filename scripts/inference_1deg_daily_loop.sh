#!/bin/bash
# Run one inference invocation per calendar day, each tightly bounded so the
# sampler can only produce a single 00Z-initialized, 96h/6h-step forecast
# (no 06/12/18Z samples, no 24h-stride substep collapse -- see
# /home/sagemaker-user/.claude/plans/iridescent-brewing-karp.md for why).
#
# Each day's raw output is written under its own run-id, then copied into the
# canonical run's results directory as an additional "rank" file. The evaluate
# reader auto-discovers all rank files for a run (rank: "all" is the default)
# and merges them into one dataset -- see
# packages/evaluate/src/weathergen/evaluate/io/wegen_reader.py:_discover_rank_files.
#
# Usage: scripts/inference_1deg_daily_loop.sh [first_day] [last_day]
#   first_day/last_day are day-of-month integers for December 2020.
#   Defaults: 2 27 (Dec 1 is unreachable due to 1-step input lookback;
#   Dec 28-31 would run past the ERA5 zarr's data cutoff of 2020-12-31T18:00).

set -euo pipefail

export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml

REPO_DIR=/home/sagemaker-user/git/WeatherGenerator
BASE_RUN_ID=era5_1deg_daily96h
FROM_RUN_ID=8level_1deg_2020_smoke2
ZARR_EXT=zip

FIRST_DAY="${1:-2}"
LAST_DAY="${2:-27}"

RESULTS_DIR="$WEATHERGEN_PRIVATE_REPO_PATH/results/$BASE_RUN_ID"
mkdir -p "$RESULTS_DIR"

rank=0
for day in $(seq -w "$FIRST_DAY" "$LAST_DAY"); do
  DAY_DATE="2020-12-${day}"
  START_DATE=$(date -d "${DAY_DATE} -6 hours" +%Y-%m-%dT%H:%M)
  END_DATE=$(date -d "${DAY_DATE} -6 hours +114 hours" +%Y-%m-%dT%H:%M)

  if [ "$rank" -eq 0 ]; then
    # First day: write directly into the canonical run-id so its run-config
    # metadata lands where the evaluate reader expects it.
    RUN_ID="$BASE_RUN_ID"
  else
    RUN_ID="${BASE_RUN_ID}_day${day}"
  fi

  echo "=== Day ${DAY_DATE} (00Z init) -> run-id=${RUN_ID}, rank=${rank}, window=[${START_DATE}, ${END_DATE}) ==="

  uv run --directory "$REPO_DIR" python scripts/monitor_resources.py \
    --interval 2 \
    --output "$WEATHERGEN_PRIVATE_REPO_PATH/results/${RUN_ID}/resource_usage_inference.csv" \
    -- \
    uv run --directory "$REPO_DIR" inference \
      --base-config=config/config_era5_2020_smoke.yml --from-run-id "$FROM_RUN_ID" \
      --run-id "$RUN_ID" \
      --options \
        test_config.start_date="${START_DATE}" \
        test_config.end_date="${END_DATE}" \
        test_config.forecast.num_steps=16 \
        data_loading.num_workers=0 \
        data_loading.memory_pinning=true

  SRC_FILE="$WEATHERGEN_PRIVATE_REPO_PATH/results/${RUN_ID}/validation_chkpt00000_rank0000.${ZARR_EXT}"
  DST_FILE=$(printf "%s/validation_chkpt00000_rank%04d.%s" "$RESULTS_DIR" "$rank" "$ZARR_EXT")

  if [ "$rank" -ne 0 ]; then
    echo "Copying ${SRC_FILE} -> ${DST_FILE}"
    cp "$SRC_FILE" "$DST_FILE"
  fi

  echo "Day ${DAY_DATE} done: rank ${rank} -> $(basename "$DST_FILE")"
  rank=$((rank + 1))
done

echo "=== Done: ${rank} daily 00Z forecasts written under ${RESULTS_DIR} ==="
echo "Next: run scripts/evaluate_1deg.sh to score/plot the merged result."
