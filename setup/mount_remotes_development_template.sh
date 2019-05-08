#!/bin/bash

sudo mkdir /mnt/afs
sudo mkdir /mnt/afs/chesapeake
sudo mkdir /mnt/afs/code
sudo chown -R $(whoami) /mnt/afs

sudo mkdir /mnt/blobfuse
sudo mkdir /mnt/blobfuse/esri-naip
sudo mkdir /mnt/blobfuse/resampled-nlcd
sudo mkdir /mnt/blobfuse/resampled-lc
sudo mkdir /mnt/blobfuse/resampled-landsat8
sudo mkdir /mnt/blobfuse/resampled-buildings

sudo mkdir /mnt/blobfuse/full-usa-output
sudo mkdir /mnt/blobfuse/middle-cedar-watershed-data

sudo mkdir /mnt/blobfuse/pred-output
sudo mkdir /mnt/blobfuse/train-output
sudo mkdir /mnt/blobfuse/cnn-minibatches

sudo chown -R $(whoami) /mnt/blobfuse

sudo mkdir /mnt/resource
sudo chown -R $(whoami) /mnt/resource

# Un-comment below lines, filling in <USERNAME> and <PASSWORD> fields
# sudo mount -t cifs //mslandcoverstorageeast.file.core.windows.net/chesapeake /mnt/afs/chesapeake -o vers=3.0,username=<USERNAME>,password=<PASSWORD>==,dir_mode=0777,file_mode=0777,serverino
# sudo mount -t cifs //msrcalebubuntu.file.core.windows.net/code               /mnt/afs/code       -o vers=3.0,username=<USERNAME>,password=<PASSWORD>==,dir_mode=0777,file_mode=0777,serverino

cd ~
cp -r /mnt/afs/chesapeake/blobfusecfgs .

# satellite data blobs
blobfuse /mnt/blobfuse/esri-naip --tmp-path=/mnt/resource/blobfusetmp  --config-file=/home/$(whoami)/blobfusecfgs/esrinaip.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
blobfuse /mnt/blobfuse/resampled-nlcd --tmp-path=/mnt/resource/nlcdblobfusetmp  --config-file=/home/$(whoami)/blobfusecfgs/nlcd.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
blobfuse /mnt/blobfuse/resampled-lc --tmp-path=/mnt/resource/lcblobfusetmp  --config-file=/home/$(whoami)/blobfusecfgs/resampled-lc.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
blobfuse /mnt/blobfuse/resampled-landsat8 --tmp-path=/mnt/resource/ls8blobfusetmp  --config-file=/home/$(whoami)/blobfusecfgs/resampled-landsat8.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
blobfuse /mnt/blobfuse/resampled-buildings --tmp-path=/mnt/resource/resampled-buildings  --config-file=/home/$(whoami)/blobfusecfgs/resampled-buildings.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

# large output blobs
blobfuse /mnt/blobfuse/full-usa-output --tmp-path=/mnt/resource/full-usa-output --config-file=/home/$(whoami)/blobfusecfgs/full-usa-output.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
blobfuse /mnt/blobfuse/middle-cedar-watershed-data --tmp-path=/mnt/resource/middle-cedar  --config-file=/home/$(whoami)/blobfusecfgs/middle-cedar-watershed-data.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120

# Le blobs
blobfuse /mnt/blobfuse/pred-output --tmp-path=/mnt/resource/pred-output  --config-file=/home/$(whoami)/blobfusecfgs/pred-output.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
blobfuse /mnt/blobfuse/train-output --tmp-path=/mnt/resource/train-output  --config-file=/home/$(whoami)/blobfusecfgs/train-output.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
blobfuse /mnt/blobfuse/cnn-minibatches --tmp-path=/mnt/resource/cnn-minibatches  --config-file=/home/$(whoami)/blobfusecfgs/cnn-minibatches.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
