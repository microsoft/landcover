#!/bin/sh
source az.config
echo "$AZ_RESOURCE_GRP"
sudo echo "deallocating vm" >> log.log
az login --identity
az vm deallocate -g $AZ_RESOURCE_GRP -n $AZ_RESOURCE_NAME
