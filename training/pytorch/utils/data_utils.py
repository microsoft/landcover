import numpy as np
import argparse
import einops
import random

parser = argparse.ArgumentParser()

parser.add_argument('--patch_files_filename', type=str, default='training/data/finetuning/val1_train_patches.txt', help='Text file containing list of npy files for patches, one per line')
parser.add_argument('--points_per_patch', type=int, nargs='+', default=[1, 2, 3, 4, 5, 10, 15, 20, 40, 60, 80, 100])

args = parser.parse_args()


def main():
    f = open(args.patch_files_filename, 'r')
    patch_file_names = [line.strip() for line in f.readlines()]
    num_patches = len(patch_file_names)

    for (i, patch_file_name) in enumerate(patch_file_names):
        print('Generating random points for patch %d' % i)
        patch = np.load(patch_file_name)
        batch_size, channel, height, width = patch.shape
        mask = np.zeros((1, len(args.points_per_patch), height, width), dtype=np.uint8)

        
        largest_num_points = max(args.points_per_patch)
        random_points = []
        while len(random_points) < 100:
            k = 92
            row = random.randint(k, height - k)
            col = random.randint(k, width - k)
            if (row, col) not in random_points:
                random_points.append((row, col))

        for l, point in enumerate(random_points):
            row, col = point
            for j, subset_size in enumerate(sorted(args.points_per_patch)):
                if l < subset_size:
                    mask[0, j:, row, col] = 1
   
        save_mask(mask, patch_file_name)


        
def save_mask(mask, patch_file_name: np.array):
    mask_file_name = patch_file_name.replace('.npy', '-mask.npy')
    np.save(mask_file_name, mask)
    

    #points_per_patch_average = (args.num_points*1.0) / num_patches
    #points_per_patch_whole = math.floor(points_per_patch_average)
    #points_accumulated = points_per_patch_whole * num_patches
    #num_missing_points = args.num_points - points_accumulated
    
    
            
            
if __name__ == '__main__':
    main()
