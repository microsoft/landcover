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

echo "gpu"
echo $CUDA_VISIBLE_DEVICES
echo "test region"
echo $TEST_REGION


#conda activate py35
#export PYTHONPATH=../../..
export PYTHONPATH=.

# Train fine-tuned model
python training/pytorch/model_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" --training_patches_fn "training/data/finetuning/test${TEST_REGION}_train_patches.txt" --log_fn "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test$TEST_REGION/finetune.csv" --model_output_directory "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}"


# Test fine-tuned models
for mask_id in {0..11}
do
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/finetuned_unet_gn.pth_[('epoch', 9), ('learning_rate', 0.03), ('lr_schedule_step_size', 5), ('mask_id', $mask_id), ('method_name', 'group_params'), ('optimizer_method', <class 'torch.optim.adam.Adam'>), ('run_id', "*")].tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}.txt
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/finetuned_unet_gn.pth_[('epoch', 9), ('last_k_layers', 1), ('learning_rate', 0.03), ('lr_schedule_step_size', 5), ('mask_id', $mask_id), ('method_name', 'last_k_layers'), ('optimizer_method', <class 'torch.optim.adam.Adam'>), ('run_id', "*")].tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}.txt
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/finetuned_unet_gn.pth_[('epoch', 9), ('last_k_layers', 2), ('learning_rate', 0.03), ('lr_schedule_step_size', 5), ('mask_id', $mask_id), ('method_name', 'last_k_layers'), ('optimizer_method', <class 'torch.optim.adam.Adam'>), ('run_id', "*")].tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}.txt
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${TEST_REGION}/finetuned_unet_gn.pth_[('epoch', 9), ('last_k_layers', 4), ('learning_rate', 0.03), ('lr_schedule_step_size', 5), ('mask_id', $mask_id), ('method_name', 'last_k_layers'), ('optimizer_method', <class 'torch.optim.adam.Adam'>), ('run_id', "*")].tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}.txt
done
    
# Test original model
python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" --test_tile_fn training/data/finetuning/test${TEST_REGION}.txt
    

