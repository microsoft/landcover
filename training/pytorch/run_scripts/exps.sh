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

#echo "gpu"
#echo $CUDA_VISIBLE_DEVICES
#echo "test region"
#echo $TEST_REGION


#conda activate py35
#export PYTHONPATH=../../..
export PYTHONPATH=.

echo "model, mean_IoU, pixel_accuracy"

num_patches=40
MODELS_DIR="/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/${num_patches}_patches"

#i=1


# Test original model
#python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/40_patches/rand_40_${i}/finetuned_unet_gn.pth_group_params_lr_0.002500_epoch_-1.tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}.txt



# Train fine-tuned model
for i in {1..5}
do
       python training/pytorch/model_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" --training_patches_fn "training/data/finetuning/test${TEST_REGION}_train_patches_rand_${num_patches}_${i}.txt" --log_fn "${MODELS_DIR}/rand_${num_patches}_${i}/train_results.csv" --model_output_directory "${MODELS_DIR}/rand_${num_patches}_${i}"


# Test fine-tuned models

    MODELS=(
	"/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/40_patches/rand_40_${i}/finetuned_unet_gn.pth_group_params_lr_0.002500_epoch_10.tar"
	"/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/40_patches/rand_40_${i}/finetuned_unet_gn.pth_last_k_layers_lr_0.015000_epoch_1_last_k_1.tar"
	"/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/40_patches/rand_40_${i}/finetuned_unet_gn.pth_last_k_layers_lr_0.000600_epoch_8_last_k_2.tar"
	"/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test1/40_patches/rand_40_${i}/finetuned_unet_gn.pth_last_k_layers_lr_0.004500_epoch_0_last_k_3.tar"
    )

    # for model_file in $(ls $MODELS_DIR/rand_${num_patches}*/*_epoch_9*)
    for model_file in ${MODELS[*]}
    do
	# echo $model_file
	python training/pytorch/test_finetuning.py --model_file "$model_file" --test_tile_fn training/data/finetuning/test${TEST_REGION}.txt
    done
done
    
    

