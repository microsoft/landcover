export CUDA_VISIBLE_DEVICES=0

for test_region in {1..2}
do

    # Train fine-tuned model
    python training/pytorch/model_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" --training_patches_fn "training/data/finetuning/test${test_region}_train_patches.txt" --log_fn "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/test/finetune.csv" --model_output_directory "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${test_region}"


    # Test fine-tuned models
    for mask_id in {0..11}
    do
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${test_region}/finetuned_unet_gn.pth_{'lr_schedule_step_size': 5, 'optimizer_method': <class 'torch.optim.adam.Adam'>, 'run_id': 1, 'method_name': 'last_k_layers', 'epoch': 9, 'last_k_layers': 1, 'learning_rate': 0.01, 'mask_id': $mask_id}.tar" --test_tile_fn training/data/finetuning/test${test_region}.txt
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${test_region}/finetuned_unet_gn.pth_{'lr_schedule_step_size': 5, 'optimizer_method': <class 'torch.optim.adam.Adam'>, 'run_id': 1, 'method_name': 'last_k_layers', 'epoch': 9, 'last_k_layers': 2, 'learning_rate': 0.01, 'mask_id': $mask_id}.tar" --test_tile_fn training/data/finetuning/test${test_region}.txt
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${test_region}/finetuned_unet_gn.pth_{'lr_schedule_step_size': 5, 'optimizer_method': <class 'torch.optim.adam.Adam'>, 'run_id': 1, 'method_name': 'last_k_layers', 'epoch': 9, 'last_k_layers': 4, 'learning_rate': 0.01, 'mask_id': $mask_id}.tar" --test_tile_fn training/data/finetuning/test${test_region}.txt
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/finetuning/test/test${test_region}/finetuned_unet_gn.pth_{'lr_schedule_step_size': 5, 'optimizer_method': <class 'torch.optim.adam.Adam'>, 'run_id': 1, 'method_name': 'group_params', 'epoch': 9, 'learning_rate': 0.03, 'mask_id': $mask_id}.tar" --test_tile_fn training/data/finetuning/test${test_region}.txt
    done
    
    # Test original model
    python training/pytorch/test_finetuning.py --model_file "/mnt/blobfuse/train-output/conditioning/models/backup_unet_gn_isotropic_nn9/training/checkpoint_best.pth.tar" --test_tile_fn training/data/finetuning/test${test_region}.txt
    

done


