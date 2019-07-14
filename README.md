# Land Cover Mapping Tool

This repository holds both the "frontend" web-application and "backend" web API server that make up our "Land Cover Mapping" tool.
An instance of this tool may be live, [here](http://msrcalebubuntu.eastus.cloudapp.azure.com:4040/).


## Setup Instructions

We develop / use the tool on Data Science Virtual Machines for Linux (Ubuntu) images on Azure (see the [Azure Portal](https://ms.portal.azure.com/)), so these setup instructions are tailored for that environment, however there is no reason that this project cannot be run on any machine.

### Initial machine setup

- Create a new VM with the Data Science Virtual Machine for Linux (Ubuntu) image via the [Azure Portal](https://ms.portal.azure.com/)
- Open the incoming ports 4040 and 4444 to the VM through the Azure Portal (these ports will be used by the web tool)
- SSH into the VM using a desktop SSH client
- Run the following commands to install the additional necessary Python packages:
```
sudo apt-get update
sudo apt-get install blobfuse
conda activate py35
conda install rasterio fiona shapely rtree
pip install --user --upgrade bottle mercantile rasterio
conda deactivate
```
- Log out and log back in
- Visit the Microsoft AI for Earth [Azure storage account](https://ms.portal.azure.com/#blade/Microsoft_Azure_Storage/FileShareMenuBlade/overview/storageAccountId/%2Fsubscriptions%2Fc9726640-cf74-4111-92f5-0d1c87564b93%2FresourceGroups%2FLandcover2%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fmslandcoverstorageeast/path/vm-fileshare) (your account will need to be given access first)
  - Download the `web-tool/mount_remotes_deployment.sh` and `web-tool/new_repo_install.sh` scripts to the home directory
  - Run the `mount_remotes_deployment.sh` script to mount the necessary blob storage containers (note: you will need to run this script every time you restart the VM)

### Repository setup instructions

- SSH into the VM using a desktop SSH client
- `git clone git@github.com:microsoft/landcover.git` (clone this repository)
- `mv new_repo_install.sh landcover/`
- `cd landcover`
- Edit `new_repo_install.sh` as appropriate. This script will copy the necessary data files from the `web-tool-data` blob container to the project directory,  however you probably don't need _all_ the data in `web-tool-data/web_tool/tiles/` as these files can be large and are project instance specific.
- `./new_repo_install.sh`
- `rm new_repo_install.sh` (to keep the project directory clean!)
- Edit `web_tool/endpoints.mine.js` and replace "msrcalebubuntu.eastus.cloudapp.azure.com" with the address of your VM (find/change your VM's host name or IP address in the Azure portal).


## Running an instance of the tool

- SSH into the VM using a desktop SSH client
- `cd landcover`
- `python web_tool/frontend_server.py` to start up a HTTP server on :4040 that will serve the actual web-application (i.e. the "frontend") and any custom basemaps in `web_tool/tiles/`.
- `python web_tool/backend_server.py --model nips_sr --model_fn web_tool/data/final_model.h5 --fine_tune last_layer --gpu 0 --debug --verbose` to start up a HTTP server on :4444 that responds to API calls from the "frontend", allowing the web-app to interface with our CNN models (i.e. the "backend").
- Note: the previous two commands will block while the servers are running. You can exeute them in seperate tmux windows to have them running even when your SSH session isn't active.
- You should now be able to visit `http://<your VM address>:4040/index.html` and see the "frontend" interface.


## Design Overview

- "Frontend"
  - `index.html`, `endpoints.js`
  - Whenever an user clicks somewhere on the map, the app will query each server defined in `endpoints.js` and show the results overlayed on the map.
  - Upon new installation, copy `endpoints.js` to `endpoints.mine.js`. This copy allows customizing the back-end server to use, and will be ignored by Git.
  - When changing the host-name and port number, the URL must end with `/` (eg. `http://msrcalebubuntu.eastus.cloudapp.azure.com:4444/`).
- "Backend"
  - Consists of `backend_server.py`, `ServerModels*.py`, `DataLoader.py`
  - `backend_server.py` starts a bottle server to serve the API
    - Can be provided a port via command line argument, must be provided a "model" to serve via command line argument.
    - The "model" that is provided via the command line argument corresponds to one of the `ServerModels*.py` files. Currently this interface is just an ugly hack.
  - `DataLoader.py` contains all the code for finding the data assosciated with a given spatial query.


## API

The "backend" server provides the following API:

### *POST `/predPatch`*

Input example:
```js
{
    "extent": { // definition of bounding box to run model on
        "xmax": bottomright.x,
        "xmin": topleft.x,
        "ymax": topleft.y,
        "ymin": bottomright.y,
        "spatialReference": {
            "latestWkid": 3857 // CRS of the coordinates
        }
    },
    "weights": [0.25, 0.25, 0.25, 0.25], // reweighting of the softmax outputs, there should be one number (per class)
}
```

Output example:
```js
{
    "extent": ..., // copied from input
    "weights": ..., // copied from input
    "model_name": "Full-US-prerun", // name of the model being served
    "input_naip": "..." // base64 encoding of input NAIP imagery used to generate the model output, as PNG
    "output_hard": "..." // base64 encoding of hard class estimates, also as PNG
    "output_soft": "..." // base64 encoding of soft class estimates, see `utils.class_prediction_to_img()` for how image is generated

}
```

### *POST `/getInput`*

Input example:
```js
{
    "extent": { // definition of bounding box to run model on
        "xmax": bottomright.x,
        "xmin": topleft.x,
        "ymax": topleft.y,
        "ymin": bottomright.y,
        "spatialReference": {
            "latestWkid": 3857 // CRS of the coordinates
        }
    },
}
```

Output example:
```js
{
    "extent": ..., // copied from input
    "input_naip": "..." // base64 encoding of input NAIP imagery used to webpagegenerate the model output, as PNG
}
```

# TODO
- Update "Design Overview" section
- Add a tutorial for using the tool
- Update the "API" section
- Explain how different datasets work
- Write section detailing the user study implementation