# Generate random patch samples
training/pytorch/utils/data/random_patch_samples.sh


PATCHES_FILE_NAME_DIR='training/data/finetuning/'
AREA_NAMES=(
    'test1'
    'test2'
    'test3'
    'test4'
    'val1'
    'val2'
)

for area_name in ${AREA_NAMES[*]}
do
    # Convert tiles from .mrf to .npy
    python training/pytorch/utils/data/tile_to_npy.py --tiles_file_name ${PATCHES_FILE_NAME_DIR}${area_name}_train_tiles.txt
    python training/pytorch/utils/data/tile_to_npy.py --tiles_file_name ${PATCHES_FILE_NAME_DIR}${area_name}_test_tiles.txt

    # Sample training patches
    # python training/pytorch/utils/data/tile_to_npy.py --tiles_file_name ${PATCHES_FILE_NAME_DIR}${area_name}_train_tiles.txt --sample --patches_output_directory /mnt/blobfuse/cnn-minibatches/summer_2019/active_learning_splits/${area_name}/
    
    # Generate random pixel masks for each patch
    python training/pytorch/utils/data/create_masks.py --patch_files_filename ${PATCHES_FILE_NAME_DIR}${area_name}_train_patches.txt

    
done

