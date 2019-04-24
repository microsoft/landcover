#!/bin/bash

# update blobfuse version
sudo apt-get update
sudo apt-get install blobfuse

# install our python geospatial library dependencies
source activate /data/anaconda/envs/py35
conda install rasterio fiona shapely rtree
source deactivate

pip install --user bottle
pip install --user einops

# sometimes the GPU doesn't work
sudo shutdown -r now
