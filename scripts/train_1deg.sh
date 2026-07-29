#!/bin/bash

export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml

RUN_ID=8level_1deg_2020_smoke2

uv run --directory /home/sagemaker-user/git/WeatherGenerator python scripts/monitor_resources.py \
  --interval 2 \
  --output "$WEATHERGEN_PRIVATE_REPO_PATH/results/${RUN_ID}/resource_usage_train.csv" \
  -- \
  uv run --directory /home/sagemaker-user/git/WeatherGenerator train \
    --base-config=config/config_era5_2020_smoke.yml --run-id "$RUN_ID"
