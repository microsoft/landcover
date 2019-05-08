#!/bin/bash

sudo mkdir /mnt/afs
sudo mkdir /mnt/afs/chesapeake
sudo chown -R $(whoami) /mnt/afs

sudo mkdir /mnt/blobfuse
sudo mkdir /mnt/blobfuse/esri-naip
sudo mkdir /mnt/blobfuse/resampled-nlcd
sudo mkdir /mnt/blobfuse/resampled-lc
sudo mkdir /mnt/blobfuse/resampled-landsat8
sudo mkdir /mnt/blobfuse/resampled-buildings

sudo mkdir /mnt/blobfuse/full-usa-output

sudo chown -R $(whoami) /mnt/blobfuse

sudo mkdir /mnt/resource
sudo chown -R $(whoami) /mnt/resource

# Un-comment below line, filling in <USERNAME> and <PASSWORD> fields
# sudo mount -t cifs //mslandcoverstorageeast.file.core.windows.net/chesapeake /mnt/afs/chesapeake -o vers=3.0,username=<USERNAME>,password=<PASSWORD>,dir_mode=0777,file_mode=0777,serverino

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
