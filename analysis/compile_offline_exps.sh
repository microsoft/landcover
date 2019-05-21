TEST_AREAS=(1, 2, 3, 4)

results_dir="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test"
analysis_dir="/mnt/blobfuse/analysis/active-learning"

mkdir ${analysis_dir}


# Entropy
# /mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/entropy/fine_tune_test_results.csv
# /mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/entropy/fine_tune_test_results_group_params.csv
for area in ${TEST_AREAS[@]}
do
    file=${analysis_dir}/test${area}/entropy/fine_tune_test_results.csv
    cat ${results_dir}/test${area}/entropy/fine_tune_test_results.csv > ${file}
    cat ${results_dir}/test${area}/entropy/fine_tune_test_results_group_params.csv >> ${file}
done


# Random
# /mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/random_step_-1/fine_tune_test_results.csv
for area in ${TEST_AREAS[@]}
do
    file=${analysis_dir}/test${area}/random/fine_tune_test_results.csv
    cp ${results_dir}/test${area}/random_step_-1/fine_tune_test_results.csv > ${file}
done


# Margin
# /mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test4/margin_step_-1/fine_tune_test_results.csv
for area in ${TEST_AREAS[@]}
do
    file=${analysis_dir}/test${area}/margin/fine_tune_test_results.csv
    cp ${results_dir}/test${area}/margin_step_-1/fine_tune_test_results.csv > ${file}
done

