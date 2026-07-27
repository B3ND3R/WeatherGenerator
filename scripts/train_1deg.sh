#!/bin/bash

export WEATHERGEN_PRIVATE_REPO_PATH=/home/sagemaker-user/weathergen_shared
export WEATHERGEN_PRIVATE_CONF=/home/sagemaker-user/.weathergen_private_conf.yml
uv run --directory /home/sagemaker-user/git/WeatherGenerator train \
  --base-config=config/config_era5_2020_smoke.yml --run-id 8level_1deg_2020_smoke2
