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

TEST_SPLITS=(
    "md_1m_2013"
    "de_1m_2013"
    "ny_1m_2013"
    "pa_1m_2013"
    "va_1m_2014"
    "wv_1m_2014"
)

((TIME_BUDGET=3600*12))
BATCH_SIZE_EXPONENT=4
((BATCH_SIZE=2**$BATCH_SIZE_EXPONENT))
GPU_ID=0
LOSS=${LOSSES[0]}
LEARNING_RATE=0.003
MODEL_TYPE=${MODEL_TYPES[5]}
NOTE="replication_1"
MODEL_FN="model_115.h5"
MODEL_FN_INST=${MODEL_FN%.*}

EXP_NAME=ForICCV-landcover-batch_size-${BATCH_SIZE}-loss-${LOSS}-lr-${LEARNING_RATE}-model-${MODEL_TYPE}-schedule-stepped-note-${NOTE}
EXP_NAME_OUT=${EXP_NAME}-instance-${MODEL_FN_INST}
OUTPUT=/mnt/blobfuse/train-output/ForICCV
PRED_OUTPUT=/mnt/blobfuse/pred-output/ForICCV


if [ ! -f "${OUTPUT}/${EXP_NAME}/${MODEL_FN}" ]; then
    echo "This experiment hasn't been trained! Exiting..."
    exit
fi


if [ -d "${PRED_OUTPUT}/${EXP_NAME_OUT}" ]; then
    echo "Experiment output ${PRED_OUTPUT}/${EXP_NAME_OUT} exists"
    while true; do
        read -p "Do you wish to overwrite this experiment? [y/n]" yn
        case $yn in
            [Yy]* ) rm -rf ${PRED_OUTPUT}/${EXP_NAME_OUT}; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer y or n.";;
        esac
    done
fi

mkdir -p ${PRED_OUTPUT}/${EXP_NAME_OUT}

echo ${MODEL_FN} > ${PRED_OUTPUT}/${EXP_NAME_OUT}/model_fn.txt

for TEST_SPLIT in "${TEST_SPLITS[@]}"
do
	echo $TEST_SPLIT
    TEST_CSV=splits/${TEST_SPLIT}_ICCV_test_split.csv
    echo ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_test_${TEST_SPLIT}.txt
    unbuffer python -u test_model_landcover.py \
        --input ${TEST_CSV} \
        --output ${PRED_OUTPUT}/${EXP_NAME_OUT}/ \
        --model ${OUTPUT}/${EXP_NAME}/${MODEL_FN} \
        --gpu ${GPU_ID} \
        &> ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_test_${TEST_SPLIT}.txt

    echo ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_acc_${TEST_SPLIT}.txt
    unbuffer python -u compute_accuracy.py \
        --input_list ${TEST_CSV} \
        --pred_blob_root ${PRED_OUTPUT}/${EXP_NAME_OUT} \
        &> ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_acc_${TEST_SPLIT}.txt &
done

wait;

echo "./eval_all_landcover_results.sh ${PRED_OUTPUT}/${EXP_NAME_OUT}"

exit 0