#!/bin/bash

ARRAY=(train_MD_region_patch_LC1_all_NLCD train_Chesapeake2013_region_patch_LC1_all_NLCD train_Chesapeake2014_region_patch_LC1_all_NLCD)

CODE=${ARRAY[${1}]}
GPU_ID=1
NUM_HIGHRES=${3}
DUPLICATION_NUM=${4}
SUPERRES=0.0
HIGHRES=40

# The distribution of data (in terms of NLCD labels) to sample from,
# in each training minibatch
TRAIN_LABEL_DIS=""\
"0_50_0.2_"\
"4_8_25_63_20_"\
"1_3_20_"\
"0_"\
"2_2_"\
"0_0_0_"\
"1.5_1.5_4_12_"\
"0"

# Name of the experiment
EXP_NAME=ForKDD-byCaleb-${CODE}-${SUPERRES}-${HIGHRES}-${NUM_HIGHRES}-${DUPLICATION_NUM}

# The list of training and validation patches
TRAIN_PATCH_LIST=/mnt/afs/chesapeake/for-le/Kolya_paper_patch_list/${CODE}.txt
TEST_PATCH_LIST=${TRAIN_PATCH_LIST}
HIGHRES_STATE=md

# Output folder
# The models, copies of the code, log, etc. will be
# saved under ${OUTPUT}/${EXP_NAME}
# Check out /mnt/blobfuse/train-output/ for existing saved enrionments
OUTPUT=/mnt/blobfuse/train-output/

mkdir -p ${OUTPUT}/${EXP_NAME}/
cp -r ./data/ *.sh *.py ${OUTPUT}/${EXP_NAME}/
echo ${OUTPUT}/${EXP_NAME}/log.txt

set -x
python -u train_model_landcover.py \
    --name=${EXP_NAME} \
    --output=${OUTPUT}/ \
    --train_patch_list=${TRAIN_PATCH_LIST} \
    --test_patch_list=${TEST_PATCH_LIST} \
    --state_to_use_highres_label=${HIGHRES_STATE} \
    --gpuid=${GPU_ID} \
    --train_label_dis=${TRAIN_LABEL_DIS} \
    --superres=${SUPERRES} \
    --highres=${HIGHRES} \
    --epochs=70 \
    &> ${OUTPUT}/${EXP_NAME}/log.txt

wait;

exit 0