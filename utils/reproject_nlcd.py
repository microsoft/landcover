import sys, os, time
import subprocess

input_fn = "/mnt/afs/code/nlcd_census/data/raw/nlcd/nlcd_2016.tif"


output_dir = "/mnt/blobfuse/web-tool-data/lsr_change_detection/tiles/la/"
target_fns = os.listdir(output_dir)
target_fns = [
    fn
    for fn in target_fns
    if "_2010.tif" in fn
]

for i, target_fn in enumerate(target_fns):
    print("%d/%d" % (i, len(target_fns)))

    output_fn = target_fn[:-8] + "2016_nlcd.tif"
    command = [
        "python", "reproject_data.py",
        "--input_fn", input_fn,
        "--target_fn", output_dir + target_fn,
        "--output_fn", output_dir + output_fn
    ]
    subprocess.call(command)
    