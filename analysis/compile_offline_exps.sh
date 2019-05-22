TEST_AREAS=(1 2 3 4)

results_dir="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test"
analysis_dir="/mnt/blobfuse/train-output/offline-active-learning"

mkdir -p ${analysis_dir}


# Entropy
# /mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/entropy/fine_tune_test_results.csv
# /mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/entropy/fine_tune_test_results_group_params.csv
for area in ${TEST_AREAS[@]}
do
    dir=${analysis_dir}/test${area}/entropy
    mkdir -p ${dir}
    file=${dir}/fine_tune_test_results.csv
    cat ${results_dir}/test${area}/entropy/fine_tune_test_results.csv > ${file}
    cat ${results_dir}/test${area}/entropy/fine_tune_test_results_group_params.csv >> ${file}
done


# Random
# /mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/random_step_-1/fine_tune_test_results.csv
for area in ${TEST_AREAS[@]}
do
    dir=${analysis_dir}/test${area}/random
    mkdir -p ${dir}
    file=${dir}/fine_tune_test_results.csv
    cat ${results_dir}/test${area}/random_step_-1/fine_tune_test_results.csv > ${file}
done


# Margin
# /mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test4/margin_step_-1/fine_tune_test_results.csv
for area in ${TEST_AREAS[@]}
do
    dir=${analysis_dir}/test${area}/margin
    mkdir -p ${dir}
    file=${dir}/fine_tune_test_results.csv
    cat ${results_dir}/test${area}/margin_step_-1/fine_tune_test_results.csv > ${file}
done

