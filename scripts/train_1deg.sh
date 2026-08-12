#!/bin/bash

export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml

RUN_ID=8level_o96_daily

TRAIN_ENTRYPOINT=train
TRAIN_ARGS=(--base-config=config/config_era5_o96_daily.yml --run-id "$RUN_ID")
RESOURCE_USAGE_FILE="resource_usage_train.csv"

if [[ "$1" == "--continue" ]]; then
  TRAIN_ENTRYPOINT=train_continue
  TRAIN_ARGS=(--from-run-id "$RUN_ID" --reuse-run-id "${TRAIN_ARGS[@]}")
  RESOURCE_USAGE_FILE="resource_usage_train_resume_$(date +%Y%m%d%H%M%S).csv"
fi

uv run --directory /home/sagemaker-user/git/WeatherGenerator python scripts/monitor_resources.py \
  --interval 2 \
  --output "$WEATHERGEN_PRIVATE_REPO_PATH/results/${RUN_ID}/${RESOURCE_USAGE_FILE}" \
  -- \
  uv run --directory /home/sagemaker-user/git/WeatherGenerator "$TRAIN_ENTRYPOINT" \
    "${TRAIN_ARGS[@]}"
