#!/usr/bin/env bash
CONF_FILE="pytorch/params/hyper_params.json"

python -u pytorch/main.py \
    --config_file ${CONF_FILE}