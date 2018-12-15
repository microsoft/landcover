# Land Cover Mapping Tool


This repository hold both the "frontend" web-application and "backend" web API server that make up our "Land Cover Mapping" demo. An instance of this demo is live, [here](http://msrcalebubuntu.eastus.cloudapp.azure.com:4040/).


## Overview

- "Frontend"
  - `index.html`, `endpoints.js`
  - Whenever an user clicks somewhere on the map, the app will query each server defined in `endpoints.js` and show the results overlayed on the map.
- "Backend"
  - Consists of `backend_server.py`, `ServerModels*.py`, `DataLoader.py`
  - `backend_server.py` starts a bottle server to serve the API
    - Can be provided a port via command line argument, must be provided a "model" to serve via command line argument.
    - The "model" that is provided via the command line argument corresponds to one of the `ServerModels*.py` files. Currently this interface is just an ugly hack.
  - `DataLoader.py` contains all the code for finding the data assosciated with a given spatial query.

## API

The "backend" server provides the following API (currently only a single endpoint):

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

## Issues/To-do list
 
- Alignment of images between clicks is off
  - I understand the root cause of this problem but haven't implemented in the backend in a way that makes the most sense yet. The ideas for the fix can be found in `DataLoader.get_naip_by_extent_fixed` -- Caleb
  - Need to make sure that the fix doesn't assume a fixed input size.
- `/predPatch` will probably _not_ work with other CRSs (besides EPSG:3857)
- `/predPatch` will probably _not_ fail in an useful way
- Clicking on the "NAIP Input" image in the web app doesn't behave as expected (it should either do nothing, or display the NAIP imagery on the map).
- We want the `backend_server.py` to be decoupled from the implementation of the code needed to run the model. The way this currently works (in `main()` of `backend_server.py`) is really hacky.
- We should probably make another API endpoint to get the input NAIP image, instead of returning it with the model output.

## Setup

Copy the files from `//mslandcoverstorageeast.file.core.windows.net/chesapeake/demo_data/` into `data/`. This should include: `list_all_naip.txt`, `tile_index.dat`, `tile_index.idx`, `tiles.p`.
