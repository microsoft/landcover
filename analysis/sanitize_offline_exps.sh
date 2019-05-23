analysis_dir="/mnt/blobfuse/train-output/offline-active-learning"

STRATEGIES=("random" "entropy" "margin" "mistake_random")
TEST_AREAS=(1 2 3 4)

header_grep_query="method, num_points"

for area in ${TEST_AREAS[@]}
do
    for strategy in ${STRATEGIES[@]}
    do
	# Get the original file here:
	file=${analysis_dir}/test${area}/${strategy}/fine_tune_test_results.csv
	
	# Put the sanitized version here (remove duplicate headers)
	sanitized=${file}.sanitized
	
	# Put just one header line into the sanitized file
	cat ${file} | grep "${header_grep_query}" | head -n 1 > ${sanitized}
	# Put actual contents, with no header lines, into sanitized file
	cat ${file} | grep -v "${header_grep_query}" | grep -v "^>" | grep -v "^->" >> ${sanitized}
	              # ^^ remove column headers         ^^ remove PDB printouts > and -> 
	# Overwrite original file with sanitized one
	mv ${sanitized} ${file}
    done
done
