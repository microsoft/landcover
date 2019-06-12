#!/usr/bin/env bash
API_KEY="api-key-here"
DIR="dir-2-save-images"

python -u pytorch/expers/planet_download/planet_downloads.py \
    --api_key ${API_KEY} \
    --directory ${DIR}
