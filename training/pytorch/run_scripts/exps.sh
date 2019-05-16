while getopts ":t:g:" opt; do
    case ${opt} in
	g )
	    CUDA_VISIBLE_DEVICES=$OPTARG
	    ;;
	t )
	    TEST_REGION=$OPTARG
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



#NUMS_PATCHES=(10 40 100 200 400 1000 2000)
NUMS_PATCHES=(400)

# Test original model
#python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/baseline_models/baseline_unet_group_params_isotropic_nn9.pth.tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}_train_tiles.txt --tile_type train
#python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/baseline_models/baseline_unet_group_params_isotropic_nn9.pth.tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}_test_tiles.txt

query_method="entropy"

for random_seed in {1..5}
do
    MODELS_DIR="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches"

    # Train fine-tuned model
    python training/pytorch/model_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" --log_fn "${MODELS_DIR}/${query_method}_${random_seed}/train_results.csv" --model_output_directory "${MODELS_DIR}/${query_method}_${random_seed}" --area test${TEST_REGION} --train_tiles_list_file_name "training/data/finetuning/test${TEST_REGION}_train_tiles.txt" --test_tiles_list_file_name "training/data/finetuning/test${TEST_REGION}_test_tiles.txt" --random_seed ${random_seed}

    # --training_patches_fn "training/data/finetuning/sampled/test${TEST_REGION}_train_patches_rand_${num_patches}_${random_seed}.txt" \
#	   --validation_patches_fn "training/data/finetuning/sampled/test${TEST_REGION}_train_patches_rand_${num_patches}_${random_seed}.txt" \

    
    
done
