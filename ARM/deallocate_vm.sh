sudo echo "deallocating vm" >> log.log
az login --identity
az vm deallocate -g landcover-vm -n landcover-vm
