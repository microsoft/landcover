#!/usr/bin/env bash
MODEL="nips_gn"
MODEL_FN="/mnt/blobfuse/train-output/conditioning/models/backup_fusionnet32_gn_8_isotropic/training/checkpoint_best.pth.tar"

python -u backend_server.py \
    --model ${MODEL} \
    --model_fn ${MODEL_FN}