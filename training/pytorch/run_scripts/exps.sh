while getopts ":t:g:s:b:" opt; do
    case ${opt} in
	g )
	    export CUDA_VISIBLE_DEVICES=$OPTARG
	    ;;
	t )
	    TEST_REGION=$OPTARG
	    ;;
	s )
	    ACTIVE_LEARNING_STRATEGY=$OPTARG
	    ;;
	b )
	    ACTIVE_LEARNING_BATCH_SIZE=$OPTARG
	    ;;
	\? )
	    echo "Invalid option: $OPTARG" 1>&2
	    ;;
	: )
	    echo "Invalid option: $OPTARG requires an argument" 1>&2
	    ;;
    esac
done
shift $((OPTIND -1))

export PYTHONPATH=.

query_method_dir="${ACTIVE_LEARNING_STRATEGY}_step_${ACTIVE_LEARNING_BATCH_SIZE}"

MODELS_DIR="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${query_method_dir}"
mkdir "${MODELS_DIR}"
RESULTS_FILE="${MODELS_DIR}/fine_tune_test_results_dropout.csv"
rm ${RESULTS_FILE}

touch $RESULTS_FILE

for random_seed in {1..5}
do
    # Train fine-tuned model
    python training/pytorch/model_finetuning.py  \
	   --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" \
	   --log_fn "${MODELS_DIR}/${random_seed}/train_results.csv" \
	   --model_output_directory "${MODELS_DIR}/${random_seed}"  \
	   --area test${TEST_REGION} \
	   --train_tiles_list_file_name "training/data/finetuning/test${TEST_REGION}_train_tiles.txt" \
	   --test_tiles_list_file_name "training/data/finetuning/test${TEST_REGION}_test_tiles.txt" \
	   --random_seed ${random_seed} \
	   --active_learning_strategy ${ACTIVE_LEARNING_STRATEGY} \
	   --active_learning_batch_size ${ACTIVE_LEARNING_BATCH_SIZE} >> ${RESULTS_FILE}
    

    # --training_patches_fn "training/data/finetuning/sampled/test${TEST_REGION}_train_patches_rand_${num_patches}_${random_seed}.txt" \
#	   --validation_patches_fn "training/data/finetuning/sampled/test${TEST_REGION}_train_patches_rand_${num_patches}_${random_seed}.txt" \

    
    
done
