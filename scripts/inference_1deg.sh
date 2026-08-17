#!/bin/bash

export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml

RUN_ID=8level_o96_daily

uv run --directory /home/sagemaker-user/git/WeatherGenerator python scripts/monitor_resources.py \
  --interval 2 \
  --output "$WEATHERGEN_PRIVATE_REPO_PATH/results/${RUN_ID}/resource_usage_inference.csv" \
  -- \
  uv run --directory /home/sagemaker-user/git/WeatherGenerator inference \
    --base-config=config/config_era5_o96_daily.yml --from-run-id 8level_o96_daily \
    --run-id "$RUN_ID" \
    --options \
      test_config.start_date=2023-01-01T00:00 \
      test_config.end_date=2023-12-31T00:00 \
      test_config.time_window_step=168:00:00 \
      test_config.forecast.num_steps=7 \
      test_config.samples_per_mini_epoch=52 \
      test_config.output.num_samples=52 \
      data_loading.num_workers=0 \
      data_loading.memory_pinning=true
