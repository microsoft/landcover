# Land Cover Mapping Tool


This repository holds both the "frontend" web-application and "backend" web API server that make up our "Land Cover Mapping" demo. An instance of this demo is live, [here](http://msrcalebubuntu.eastus.cloudapp.azure.com:4040/).


## Setup Instructions

- Create a new Deep Learning Virtual Machine (DLVM) Ubuntu image via [Azure Portal](https://ms.portal.azure.com/)
  - Region: "East US"
  - Size:
    - "NC6s_v3  Standard GPU" (if training/running models)
    - "A4v2" (if simply running scripts that need access to AI for Earth storage accounts)
  - Image: Data Science Virtual Machine (Ubuntu)
- SSH into the virtual machine using a desktop SSH client
- Run `git clone git@github.com:microsoft/landcover.git`
- Run `cd landcover`
- Visit the Microsoft AI for Earth [Azure storage account](https://ms.portal.azure.com/#blade/Microsoft_Azure_Storage/FileShareMenuBlade/overview/storageAccountId/%2Fsubscriptions%2Fc9726640-cf74-4111-92f5-0d1c87564b93%2FresourceGroups%2FLandcover2%2Fproviders%2FMicrosoft.Storage%2FstorageAccounts%2Fmslandcoverstorageeast/path/vm-fileshare) (your account will need to be given access first)
  - Download `mount_remotes_development.sh` and `mount_remotes_deployment.sh` to `~/landcover/setup/` in your VM
  - Do NOT commit these files to Git -- though they should be ignored anyway via .gitignore
- Run `setup/new_vm_setup.sh`, this will restart the machine at the end as I have faced GPU problems on newly provisioned DLVM image machines
- Log in to VM again
- Run: `cd landcover`
- Run: `setup/after_restart.sh` (you must re-run this command any time the VM has been shut down)
- Run: `source setup/login.sh` (you must re-run this any time you log in to the VM again, even if it has not been shut down)

- Run: `cp web_tool/endpoints.js web_tool/endpoints.mine.js`
- Edit `endpoints.mine.js` (eg., run `nano endpoints.mine.js`) to point to your own server URL (find your VM's host name or IP address in the Azure portal); indicating ports for whichever backend server.py instances you are running (4444 is the default, but you can set alternate ports from command line flags to have multiple servers running, see below)

- To run the servers:
  - Open up ports 4040 and 4444 to the machine through [Azure Portal](https://ms.portal.azure.com/)
  - Run `python web_tool/frontend_server.py` this will start up a HTTP server on :4040 to serve the actual webpage
  - Run `python web_tool/backend_server.py --port 4444 --model nips_sr --fine_tune last_layer --model_fn /mnt/blobfuse/train-output/ForICCV/ForICCV-landcover-batch_size-16-loss-superres-lr-0.003-model-unet2-schedule-stepped-note-replication_1/final_model.h5 --gpu 0 --verbose` will start up a HTTP server on :4444 that serves our precomputed results with the documented API
  - alternatively use --model 2 to serve results that are computed from a CNTK model
  - You may now visit `<VM_name_or_IP>:4040` in a browser, where VM host name or IP address is the same as you logged in to via SSH, and can be found in the Azure Portal


## Code Overview

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
    "input_naip": "..." // base64 encoding of input NAIP imagery used to generate the model output, as PNG
}
```




## Issues/To-do list

- `/predPatch` will probably _not_ work with other CRSs (besides EPSG:3857)
- `/predPatch` will probably _not_ fail in an useful way
- We want the `backend_server.py` to be decoupled from the implementation of the code needed to run the model. The way this currently works (in `main()` of `backend_server.py`) is really hacky.
- If you switch the "Sharpness" slider immediately after clicking on the map (before results are returned) then an error happens.
