# Land Cover Mapping API


This folder contains code that can be used to create the landcover api (to be deployed using docker), built using AI for Earth the base image 
( Refer to [AI for Earth API Development tutorial](https://github.com/Microsoft/AIforEarth-API-Development/blob/master/Quickstart.md ) for more info)

## Overview

- 'api folder'
  - `runserver.py` - contains post and get methods for querying the API
- 'DockerFile'
  - Contains command line instructions used by docker to build a docker image, it also includes commands for installing packages required
    for the api
- 'blob_mount.json'
    - Contains information (account name, account key, container name) that will be mounted on the server when docker run command
      is executed

## API

Contains the follwing methods:

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


## Setup
1.  Copy the files from `//mslandcoverstorageeast.file.core.windows.net/chesapeake/demo_data/` into `data/`. This should include: `list_all_naip.txt`, `tile_index.dat`, `tile_index.idx`, `tiles.p`.
2.  Update blob_mount.json file with the account key for blob storage 'mslandcoverstorageeast'
3.  Navigate to the folder containing the 'DockerFile' then build the image using docker using the following command  (Refer to [AI for Earth API Development tutorial](https://github.com/Microsoft/AIforEarth-API-Development/blob/master/Quickstart.md) for more info):

  ```
  docker build . -t your_custom_image_name:1
  ```
4.  Run the image by executing the following command (Refer to [AI for Earth API Development tutorial](https://github.com/Microsoft/AIforEarth-API-Development/blob/master/Quickstart.md) for more info):

  ```
  docker run -p 8001:80 --privileged --cap-add=SYS_ADMIN --device=/dev/fuse "your_custom_image_name:1"
  ```
  If you get an error port 8000 is in use try with a different port number 
  
