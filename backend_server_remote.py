#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import sys
import os
import bottle
import argparse
import functools
import base64
import json

import numpy as np
import cv2

import DataManager
import utils

import requests

def enable_cors():
    '''From https://gist.github.com/richard-flosi/3789163

    This globally enables Cross-Origin Resource Sharing (CORS) headers for every response from this server.
    '''
    bottle.response.headers['Access-Control-Allow-Origin'] = '*'
    bottle.response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    bottle.response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

def do_options():
    '''This method is necessary for CORS to work (I think --Caleb)
    '''
    bottle.response.status = 204
    return

def pred_patch(model):
    ''' TODO: What are we doing
    '''
    bottle.response.content_type = 'application/json'

    # Inputs
    data = bottle.request.json
    extent = data["extent"]
    weights = np.array(data["weights"], dtype=np.float32)

    # ------------------------------------------------------
    # Step 1
    #   Transform the input extent into a shapely geometry
    #   Find the tile assosciated with the geometry
    # ------------------------------------------------------
    geom = DataManager.extent_to_transformed_geom(extent)
    # OLD VERSION
    '''
    return_code, return_msg = DataManager.lookup_tile_by_geom(geom)
    # TODO: How to handle errors
    if not return_code:
        bottle.response.status = 400
        return json.dumps({"error": return_msg})
    else:
        tile_fn = return_msg
    '''

    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------

    # OLD VERSION
    #naip_data, pat_trans, pat_crs, pat_bounds, padding = DataLoader.get_naip_by_extent(tile_fn, extent)
    #naip_data = np.rollaxis(naip_data, 0, 3)padding
    # sending get request and saving the response as response object 
    URL = "http://msrcalebubuntu.eastus.cloudapp.azure.com:4444/getInput"
    PARAMS = {
        "extent": data["extent"]
    }
    r = requests.post(url=URL, json=PARAMS)
    response = r.json()
    naip_data = np.fromstring(base64.b64decode(response["input_naip"]), np.uint8)
    naip_data = cv2.imdecode(naip_data, cv2.IMREAD_COLOR)
    padding = 0
    name = "Testing"

    #landsat_data = DataLoader.get_landsat_by_extent(tile_fn, extent, padding)
    #landsat_data = np.rollaxis(landsat_data, 0, 3)
    
    #nlcd_data = DataLoader.get_nlcd_by_extent(tile_fn, extent, padding)
    #nlcd_data = np.rollaxis(to_one_hot(nlcd_data, 22), 0, 3)
    
    #lc_data = DataLoader.get_lc_by_extent(tile_fn, extent, padding)
    #lc_data = np.rollaxis(to_one_hot(lc_data, 7), 0, 3)

    #blg_data = DataLoader.get_blg_by_extent(tile_fn, extent, padding)
    #blg_data = np.rollaxis(blg_data, 0, 3)
    
    # ------------------------------------------------------
    # Step 3
    #   Run a model on the input data
    #   Apply reweighting
    #   Fix padding
    # ------------------------------------------------------
    
    ## TODO Anthony: Run your model here
    # OLD VERSION
    #output, name = model(naip_data, tile_fn, extent, padding)
    output = np.random.rand(naip_data.shape[0], naip_data.shape[0], 4)
    def softmax(output):
        output_max = np.max(output, axis=2, keepdims=True)
        exps = np.exp(output-output_max)
        exp_sums = np.sum(exps, axis=2, keepdims=True)
        return exps/exp_sums
    output = softmax(output)

    assert output.shape[2] == 4, "The model function should return an image shaped as (height, width, num_classes)"
    output *= weights[np.newaxis, np.newaxis, :] # multiply by the weight vector
    sum_vals = output.sum(axis=2) # need to normalize sums to 1 in order for the rendered output to be correct
    output = output / (sum_vals[:,:,np.newaxis] + 0.000001)


    output_save = output.copy() # keep the original output to save later
    if padding > 0:
        output = output[padding:-padding, padding:-padding, :]


    # ------------------------------------------------------
    # Step 4
    #   Convert images to base64 and return  
    # ------------------------------------------------------
    img_soft = np.round(utils.class_prediction_to_img(output, False)*255,0).astype(np.uint8)
    img_soft = cv2.imencode(".png", cv2.cvtColor(img_soft, cv2.COLOR_RGB2BGR))[1].tostring()
    img_soft = base64.b64encode(img_soft).decode("utf-8")
    data["output_soft"] = img_soft

    img_hard = np.round(utils.class_prediction_to_img(output, True)*255,0).astype(np.uint8)
    img_hard = cv2.imencode(".png", cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGR))[1].tostring()
    img_hard = base64.b64encode(img_hard).decode("utf-8")
    data["output_hard"] = img_hard

    data["model_name"] = name

    bottle.response.status = 200
    return json.dumps(data)

def get_input():
    ''' TODO: What are we doing
    '''
    bottle.response.content_type = 'application/json'

    # Inputs
    data = bottle.request.json
    
    bottle.response.status = 501 # not implemented
    return ""

def do_get():
    '''Dummy method for easily testing whether the server is running correctly'''
    return "Backend server running"

def main():
    parser = argparse.ArgumentParser(description="Backend Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4444)
    parser.add_argument("--model", action="store", dest="model", choices=["1","2","3"], help="Model to use", required=True)

    args = parser.parse_args(sys.argv[1:])


    # Here we dynamically load a method that will execute whatever model we want to run when someone calls `/predPatch`
    ''' NOTE: If you want to implement new models to incorporate with this code, they should be added below.
    TODO: This "run_model" method signature should be standardized.
    '''
    ## TODO Anthony: Add an argument to start up the server with your model
    loaded_model = None
    if args.model == "1":
        import ServerModelsCached
        loaded_model = ServerModelsCached.run
    elif args.model == "2":
        import ServerModelsICLR
        loaded_model = ServerModelsICLR.run
    else:
        print("Model isn't implemented, aborting")
        return
    # We pass the dynamically loaded method to the `predPatch` callback as an argument 
    custom_pred_patch = functools.partial(pred_patch, model=loaded_model)


    # Setup the bottle server 
    app = bottle.Bottle()

    app.add_hook("after_request", enable_cors)
    app.route("/predPatch", method="OPTIONS", callback=do_options)
    app.route('/predPatch', method="POST", callback=custom_pred_patch)
    
    app.route("/getInput", method="OPTIONS", callback=do_options)
    app.route('/getInput', method="POST", callback=get_input)

    app.route('/', method="GET", callback=do_get)

    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": args.verbose,
        "server": "tornado",
        "reloader": False # Every time we change something the server will automatically reload. This breaks CNTK.
    }
    app.run(**bottle_server_kwargs)

if __name__ == '__main__':
    main()
