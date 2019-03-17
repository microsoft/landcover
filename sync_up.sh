#!/bin/bash


rsync -vzc --no-owner --no-group --omit-dir-times --ignore-times --files-from=files_to_sync.txt . caleb@msrcalebubuntu.eastus.cloudapp.azure.com:land-cover-mapping/

#rsync -rvzcn --exclude-from=files_not_to_sync.txt msrcalebubuntu.eastus.cloudapp.azure.com:land-cover-mapping/data/ data/


#scp data/training_set_1.geojson msrcalebubuntu.eastus.cloudapp.azure.com:land-cover-mapping/data/
