TEST_REGIONS=(1 2 3 4)
NUMS_PATCHES=(2 4 6 8 10 40 100 200 400 1000 2000)
RANDOM_SEEDS=(1 2 3 4 5)

for region in ${TEST_REGIONS[*]}
do
    for num_patches in ${NUMS_PATCHES[*]}
    do
	for random_seed in ${RANDOM_SEEDS[*]}
	do
	    shuf -n $num_patches training/data/finetuning/test${region}_train_patches.txt > training/data/finetuning/sampled/test${region}_train_patches_rand_${num_patches}_${random_seed}.txt
	done
    done
done

