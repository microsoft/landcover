resource_grp= $1
vm_name = $2
sudo echo "deallocating vm" >> log.log
az login --identity
az vm deallocate -g $resource_grp -n $vm_name
