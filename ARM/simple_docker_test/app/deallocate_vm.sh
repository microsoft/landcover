resource_grp= $az_resource_grp
vm_name = $az_vm_name
sudo echo "deallocating vm" >> log.log
az login --identity
az vm deallocate -g $resource_grp -n $vm_name
