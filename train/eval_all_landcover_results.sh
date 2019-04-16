#!/bin/bash

TEST_SPLITS=(
    "md_1m_2013"
    "de_1m_2013"
    "ny_1m_2013"
    "pa_1m_2013"
    "va_1m_2014"
    "wv_1m_2014"
)
EXP_BASE=${1}


echo "Test results for ${EXP_BASE}:"
echo "---------------------------------"
for TEST_SPLIT in "${TEST_SPLITS[@]}"
do

    echo "${TEST_SPLIT}"
    ./eval_landcover_results.sh ${EXP_BASE}/log_acc_${TEST_SPLIT}.txt
done

#echo "log_acc_MD_2013"
#./eval_landcover_results.sh /mnt/blobfuse/pred-output/${EXP_BASE}/log_acc_MD_2013.txt
#echo "log_acc_Chesapeake_2013"
#./eval_landcover_results.sh /mnt/blobfuse/pred-output/${EXP_BASE}/log_acc_Chesapeake_2013.txt
#echo "log_acc_Chesapeake_2014"
#./eval_landcover_results.sh /mnt/blobfuse/pred-output/${EXP_BASE}/log_acc_Chesapeake_2014.txt