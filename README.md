# Land Cover Mapping Tool

This repository holds both the "frontend" web-application and "backend" web API server that make up our "Land Cover Mapping" tool.
An instance of this tool may be live [here](http://aka.ms/landcoverdemo).


# Setup

The following sections describe how to setup the dependencies (python packages and demo data) for the "web-tool/" component of this project. We develop / use this tool on Data Science Virtual Machines for Linux (through Azure) in conjunction with specific AI for Earth projects, so the first set of instructions - "Azure VM setup instructions" - are specific for recreating our internal development environment. The second set of instructions - "Local setup instructions" - should apply more broadly.

## Azure VM setup instructions

We develop / use the tool on Data Science Virtual Machines for Linux (Ubuntu) images on Azure (see the [Azure Portal](https://portal.azure.com/)), so these setup instructions are tailored for that environment, however there is no reason that this project cannot be run on any machine (see "Local setup instructions below").

### Initial machine setup

- Create a new VM with the Data Science Virtual Machine for Linux (Ubuntu) image via the [Azure Portal](https://ms.portal.azure.com/)
- Open the incoming port 4444 to the VM through the Azure Portal
- SSH into the VM using a desktop SSH client
- Run the following commands to install the additional necessary Python packages:
```bash
sudo apt-get update
sudo apt-get install blobfuse

conda config --set channel_priority strict
conda create -y -n ai4e python=3.6
## make sure `which python` points to the python installation in our new environment
conda deactivate
conda activate ai4e
conda install -y -c conda-forge gdal rasterio fiona shapely opencv rtree
```
- Log out and log back in
- Visit the Microsoft AI for Earth [Azure storage account](https://portal.azure.com/#blade/Microsoft_Azure_Storage/FileShareMenuBlade/overview/storageAccountId/%2Fsubscriptions%2Fc9726640-cf74-4111-92f5-0d1c87564b93%2FresourceGroups%2FLandcover2%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fmslandcoverstorageeast/path/vm-fileshare) (your account will need to be given access first)
  - Download the `web-tool/mount_remotes_for_deployment.sh` and `web-tool/web_tool_data_install.sh` scripts to the home directory of the VM
  - Run the `mount_remotes_for_deployment.sh` script to mount the necessary blob storage containers (note: you will need to run this script every time you restart the VM)

### Repository setup instructions

- SSH into the VM using a desktop SSH client
```bash
# Get this repository
git clone git@github.com:microsoft/landcover.git

mv web_tool_data_install.sh landcover/
cd landcover
# Edit `web_tool_data_install.sh` as appropriate. This script will copy the necessary data files from the `web-tool-data` blob container to the project directory,  however you probably don't need _all_ the data in `web-tool-data/web_tool/tiles/` as these files can be large and are project instance specific.
bash web_tool_data_install.sh
rm web_tool_data_install.sh
cd ~

# install the project required files
cd landcover/
python -m pip install -r requirements.txt
cd ~

# Finally, setup and run the server using the demo model
cd landcover
cp web_tool/endpoints.js web_tool/endpoints.mine.js
## Edit `web_tool/endpoints.mine.js` and replace "msrcalebubuntu.eastus.cloudapp.azure.com" with the address of your machine (find/change your VM's host name or IP address in the Azure portal)
nano web_tool/endpoints.mine.js

## Edit `self._WORKERS` of the SessionHandler class in SessionHandler.py to include the GPU resources you want to use on your machine. By default this is set to use GPU IDs 0 through 4. Check `nvidia-smi` to see GPU information.
nano web_tool/SessionHandler.py
cd ~
```

## Local setup instructions

### Initial machine setup

- Make sure the incoming port 4444 is open
- Open a terminal on the machine 
- Run the following commands to install the additional necessary packages:
```bash
# Install Anaconda
cd ~
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh # select "yes" for setting up conda init
rm Anaconda3-2019.07-Linux-x86_64.sh

## logout and log back in
exit

# Install CUDA if needed; note this may require a reboot
## https://www.tensorflow.org/install/gpu#install_cuda_with_apt

# Install unzip and a library that opencv will need
sudo apt update
sudo apt install -y unzip libgl1

# Create a new conda environment for running the web tool
## setting strict channel_priority seems to be a very important step - else all the gdal dependencies are very broken
conda config --set channel_priority strict
conda create -y -n ai4e python=3.6
## make sure `which python` points to the python installation in our new environment
conda deactivate
conda activate ai4e
conda install -y -c conda-forge gdal rasterio fiona shapely opencv rtree
``` 

### Repository setup instructions

```bash
# Get the project and demo project data
git clone https://github.com/microsoft/landcover.git

wget -O landcover.zip "https://www.dropbox.com/s/tyh5qo8edog9vdh/landcover.zip?dl=1"
unzip landcover.zip
rm landcover.zip

# unzip the tileset that comes with the demo data 
cd landcover/web_tool/tiles/
unzip -q hcmc_sentinel_tiles.zip
cd ~

# install the project required files
cd landcover/
python -m pip install -r requirements.txt
cd ~

# Finally, setup and run the server using the demo model
cd landcover
cp web_tool/endpoints.js web_tool/endpoints.mine.js
## Edit `web_tool/endpoints.mine.js` and replace "msrcalebubuntu.eastus.cloudapp.azure.com" with the address of your machine
nano web_tool/endpoints.mine.js

## Edit `self._WORKERS` of the SessionHandler class in SessionHandler.py to include the GPU resources you want to use on your machine. By default this is set to use GPU IDs 0 through 4.
nano web_tool/SessionHandler.py
cd ~
```

# Running an instance of the tool

Whether you setup the server in an Azure VM or locally, the following steps should apply to start an instance of the server:
- Open a terminal on the machine that you setup (e.g. SSH into the VM using a desktop SSH client)
- `cd landcover`
- `export PYTHONPATH=.`
- `python web_tool/server.py --port 4444 --storage_type file --storage_path test.csv local`
  - This will start an HTTP server on :4444 that both serves the "frontend" web application and responds to API calls from the "frontend", allowing the web-app to interface with our CNN models (i.e. the "backend").
  - The tool comes preloaded with a dataset (defined in `web_tool/datasets.json`) and two models (defined in `web_tool/models.json`).
- You should now be able to visit `http://<your machine's address>:4444/` and see the "frontend" interface.