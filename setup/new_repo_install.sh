#!/bin/bash

# Deal with data
cd web_tool
cp -r /mnt/afs/chesapeake/demo_data/ data/
mkdir tiles/
mv data/tiles.zip tiles/
mv data/demo_set_1.zip tiles/
cd tiles
#unzip tiles.zip
unzip demo_set_1.zip
cd ..

