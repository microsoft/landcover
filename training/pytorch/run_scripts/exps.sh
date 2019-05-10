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

#TEST_LOG_FILE="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/results.csv"
echo "model, area, tile_type, tile_index, mean_IoU, pixel_accuracy, tile_path, predictions_path"

NUMS_PATCHES=(2 4 6 8 10 40 100 200 400 1000 2000)


# Test original model
python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/baseline_models/baseline_unet_group_params_isotropic_nn9.pth.tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}_train_tiles.txt --tile_type train
python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/baseline_models/baseline_unet_group_params_isotropic_nn9.pth.tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}_test_tiles.txt


for num_patches in ${NUMS_PATCHES[*]}
do
num_patches=40

    MODELS_DIR="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches"

    random_seed=1
    for random_seed in {1..5}
		       
    do
	# Train fine-tuned model
	python training/pytorch/model_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" --training_patches_fn "training/data/finetuning/test${TEST_REGION}_train_patches_rand_${num_patches}_${random_seed}.txt" --log_fn "${MODELS_DIR}/rand_${num_patches}_${random_seed}/train_results.csv" --model_output_directory "${MODELS_DIR}/rand_${num_patches}_${random_seed}"
    
	# Test fine-tuned models
	MODELS=(
	    "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches/rand_${num_patches}_${random_seed}/finetuned_unet_gn.pth_group_params_lr_0.002500_epoch_10.tar"
	    "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches/rand_${num_patches}_${random_seed}/finetuned_unet_gn.pth_last_k_layers_lr_0.015000_epoch_1_last_k_1.tar"
	    "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches/rand_${num_patches}_${random_seed}/finetuned_unet_gn.pth_last_k_layers_lr_0.000600_epoch_8_last_k_2.tar"
	    "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches/rand_${num_patches}_${random_seed}/finetuned_unet_gn.pth_last_k_layers_lr_0.004500_epoch_0_last_k_3.tar"
	)

	for model_file in ${MODELS[*]}
	do
	    python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "$model_file" --test_tile_fn training/data/finetuning/test${TEST_REGION}_test_tiles.txt --tile_type test
	    python training/pytorch/test_finetuning.py --area test${TEST_REGION} --model_file "$model_file" --test_tile_fn training/data/finetuning/test${TEST_REGION}_train_tiles.txt --tile_type train
	done
    done
done

