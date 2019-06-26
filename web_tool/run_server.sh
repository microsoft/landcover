#!/usr/bin/env bash
MODEL="group_norm"
FINE_TUNE="group_params_then_last_k"
MODEL_FN="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar"
#MODEL_FN="/mnt/blobfuse/train-output/conditioning/models/backup_fusionnet32_gn_8_isotropic/training/checkpoint_best.pth.tar"

python -u web_tool/backend_server.py \
       --model ${MODEL} \
       --fine_tune ${FINE_TUNE} \
       --model_fn ${MODEL_FN}
