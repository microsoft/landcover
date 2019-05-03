#!/bin/bash

# update blobfuse version
sudo apt-get update
sudo apt-get install blobfuse

# install our python geospatial library dependencies
source activate /data/anaconda/envs/py35
conda install rasterio fiona shapely rtree
pip install --user bottle
pip install --user einops
source deactivate


# Deal with data
./after_restart.sh
cd web_tool
cp -r /mnt/afs/chesapeake/demo_data/ data/
mkdir tiles/
cp data/tiles.zip tiles/
cp data/demo_set_1.zip tiles/
cd tiles
unzip tiles.zip
unzip demo_set_1.zip
cd ..

# sometimes the GPU doesn't work
sudo shutdown -r now
