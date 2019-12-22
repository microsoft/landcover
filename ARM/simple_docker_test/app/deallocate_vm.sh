. az_config.cfg
sudo echo "deallocating vm" >> log.log
az login --identity
az vm deallocate -g $az_resource_grp -n $az_vm_name
