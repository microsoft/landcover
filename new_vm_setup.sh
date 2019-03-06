#!/bin/bash

# update blobfuse version
sudo apt-get install blobfuse

# install our python geospatial library dependencies
source activate py35
conda install rasterio fiona shapely rtree
source deactivate

pip install --user bottle

# sometimes the GPU doesn't work
sudo shutdown -r now