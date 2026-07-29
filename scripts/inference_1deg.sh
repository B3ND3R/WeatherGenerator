#!/bin/bash

export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml

RUN_ID=era5_1deg_daily96h

uv run --directory /home/sagemaker-user/git/WeatherGenerator python scripts/monitor_resources.py \
  --interval 2 \
  --output "$WEATHERGEN_PRIVATE_REPO_PATH/results/${RUN_ID}/resource_usage_inference.csv" \
  -- \
  uv run --directory /home/sagemaker-user/git/WeatherGenerator inference \
    --base-config=config/config_era5_2020_smoke.yml --from-run-id 8level_1deg_2020_smoke2 \
    --run-id "$RUN_ID" \
    --options \
      test_config.time_window_step=24:00:00 \
      test_config.forecast.num_steps=16 \
      test_config.samples_per_mini_epoch=60 \
      data_loading.num_workers=0 \
      data_loading.memory_pinning=true
