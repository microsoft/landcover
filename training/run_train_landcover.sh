#!/bin/bash

LOSSES=(
    "crossentropy"
    "jaccard"
    "superres"
)

MODEL_TYPES=(
    "baseline"
    "extended"
    "extended_bn"
    "extended2_bn"
    "unet1"
    "unet2"
)

((TIME_BUDGET=3600*12))
BATCH_SIZE_EXPONENT=4
((BATCH_SIZE=2**$BATCH_SIZE_EXPONENT))
GPU_ID=3
LOSS=${LOSSES[0]}
LEARNING_RATE=0.003
MODEL_TYPE=${MODEL_TYPES[5]}
NOTE="replication_1"

TRAIN_PATCH_LIST=data/md_1m_2013_train_patches.txt
#TRAIN_PATCH_LIST=data/md+ny_1m_2013_train_patches.txt
SUPERRES_STATE_LIST="ny_1m_2013"
VAL_PATCH_LIST=data/md_1m_2013_val_patches.txt

EXP_NAME=ForICCV-landcover-batch_size-${BATCH_SIZE}-loss-${LOSS}-lr-${LEARNING_RATE}-model-${MODEL_TYPE}-schedule-stepped-note-${NOTE}
OUTPUT=/mnt/blobfuse/train-output/ForICCV

if [ -d "${OUTPUT}/${EXP_NAME}" ]; then
    echo "Experiment ${OUTPUT}/${EXP_NAME} exists"
    while true; do
        read -p "Do you wish to overwrite this experiment? [y/n]" yn
        case $yn in
            [Yy]* ) rm -rf ${OUTPUT}/${EXP_NAME}; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer y or n.";;
        esac
    done
fi

mkdir -p ${OUTPUT}/${EXP_NAME}/
cp -r *.sh *.py ${OUTPUT}/${EXP_NAME}/

echo ${OUTPUT}/${EXP_NAME}/log.txt

unbuffer python -u train_model_landcover.py \
    --name ${EXP_NAME} \
    --output ${OUTPUT} \
    --gpu ${GPU_ID} \
    --model_type ${MODEL_TYPE} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --loss ${LOSS} \
    --time_budget ${TIME_BUDGET} \
    --verbose 1 \
    --training_patches ${TRAIN_PATCH_LIST} \
    --validation_patches ${VAL_PATCH_LIST} \
    --superres_states ${SUPERRES_STATE_LIST} \
   &> ${OUTPUT}/${EXP_NAME}/log.txt &

# python -u train_model_landcover.py \
#     --name ${EXP_NAME} \
#     --output ${OUTPUT} \
#     --gpu ${GPU_ID} \
#     --model_type ${MODEL_TYPE} \
#     --learning_rate ${LEARNING_RATE} \
#     --batch_size ${BATCH_SIZE} \
#     --loss ${LOSS} \
#     --time_budget ${TIME_BUDGET} \
#     --verbose 1 \
#     --training_patches ${TRAIN_PATCH_LIST} \
#     --validation_patches ${VAL_PATCH_LIST} \
#     --superres_states ${SUPERRES_STATE_LIST}

#wait;
exit

#done
