vm_name = $1
resource_grp = $2
sudo echo "deallocating vm" >> log.log
az login --identity
az vm deallocate -g $vm_name -n $resource_grp
