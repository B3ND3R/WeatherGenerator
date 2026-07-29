#!/bin/bash

export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml

# Must match the run_ids entry evaluated in config/evaluate/eval_test.yml
RUN_ID=era5_1deg_daily96h

uv run --directory /home/sagemaker-user/git/WeatherGenerator python scripts/monitor_resources.py \
  --interval 2 \
  --output "$WEATHERGEN_PRIVATE_REPO_PATH/results/${RUN_ID}/resource_usage_evaluate.csv" \
  -- \
  uv run --directory /home/sagemaker-user/git/WeatherGenerator evaluate \
    --config=config/evaluate/eval_test.yml
