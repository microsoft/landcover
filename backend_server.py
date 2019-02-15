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

import DataLoader
import GeoTools
import utils

import ServerModelsICLRFormat, ServerModelsCachedFormat

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
    ''' Method called for POST `/predPatch`

    `model` is a method created in main() based on the `--model` command line argument
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
    geom = GeoTools.extent_to_transformed_geom(extent, "EPSG:4269")
    try:
        naip_fn = DataLoader.lookup_tile_by_geom(geom)
    except ValueError as e:
        print(e)
        bottle.response.status = 400
        return json.dumps({"error": str(e)})

    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------

    naip_data, padding = DataLoader.get_data_by_extent(naip_fn, extent, DataLoader.GeoDataTypes.NAIP)
    naip_data = np.rollaxis(naip_data, 0, 3)
    
    #landsat_data = DataLoader.get_landsat_by_extent(naip_fn, extent, padding)
    #landsat_data = np.rollaxis(landsat_data, 0, 3)
    
    #nlcd_data = DataLoader.get_nlcd_by_extent(naip_fn, extent, padding)
    #nlcd_data = np.rollaxis(to_one_hot(nlcd_data, 22), 0, 3)
    #nlcd_data = np.squeeze(nlcd_data)
    #nlcd_data = np.vectorize(utils.NLCD_CLASS_TO_IDX.__getitem__)(nlcd_data)

    #lc_data = DataLoader.get_lc_by_extent(naip_fn, extent, padding)
    #lc_data = np.rollaxis(to_one_hot(lc_data, 7), 0, 3)

    #blg_data = DataLoader.get_blg_by_extent(naip_fn, extent, padding)
    #blg_data = np.rollaxis(blg_data, 0, 3)
    
    # ------------------------------------------------------
    # Step 3
    #   Run a model on the input data
    #   Apply reweighting
    #   Fix padding
    # ------------------------------------------------------
    #output, name = ServerModels_Baseline_Blg_test.run_cnn(naip_data, landsat_data, blg_data, with_smooth=False)
    #name += "_with_smooth_False"
    output, name = model.run(naip_data, naip_fn, extent, padding)
    assert output.shape[2] == 4, "The model function should return an image shaped as (height, width, num_classes)"
    output *= weights[np.newaxis, np.newaxis, :] # multiply by the weight vector
    sum_vals = output.sum(axis=2) # need to normalize sums to 1 in order for the rendered output to be correct
    output = output / (sum_vals[:,:,np.newaxis] + 0.000001)
    
    if padding > 0:
        output = output[padding:-padding,padding:-padding,:]

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
    ''' Method called for POST `/getInput`
    '''
    bottle.response.content_type = 'application/json'

    # Inputs
    data = bottle.request.json
    extent = data["extent"]

    # ------------------------------------------------------
    # Step 1
    #   Transform the input extent into a shapely geometry
    #   Find the tile assosciated with the geometry
    # ------------------------------------------------------
    geom = GeoTools.extent_to_transformed_geom(extent, "EPSG:4269")
    try:
        naip_fn = DataLoader.lookup_tile_by_geom(geom)
    except ValueError as e:
        print(e)
        bottle.response.status = 400
        return json.dumps({"error": str(e)})

    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------

    naip_data, padding = DataLoader.get_data_by_extent(naip_fn, extent, DataLoader.GeoDataTypes.NAIP)
    naip_data = np.rollaxis(naip_data, 0, 3)
    naip_img = naip_data[:,:,:3].copy().astype(np.uint8) # keep the RGB channels to save as a color image later
    if padding > 0:
        naip_img = naip_img[padding:-padding,padding:-padding,:]

    img_naip = cv2.imencode(".png", cv2.cvtColor(naip_img, cv2.COLOR_RGB2BGR))[1].tostring()
    img_naip = base64.b64encode(img_naip).decode("utf-8")
    data["input_naip"] = img_naip

    bottle.response.status = 200
    return json.dumps(data)

def do_get():
    '''Dummy method for easily testing whether the server is running correctly'''
    return "Backend server running"

def main():
    parser = argparse.ArgumentParser(description="Backend Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4444)
    parser.add_argument("--model", action="store", dest="model", choices=["cached", "keras"], help="Model to use", required=True)
    parser.add_argument("--model_fn", action="store", dest="model_fn", type=str, help="Model fn to use", default=None)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", default=0)

    args = parser.parse_args(sys.argv[1:])


    # Here we dynamically load a method that will execute whatever model we want to run when someone calls `/predPatch`
    ''' NOTE: If you want to implement new models to incorporate with this code, they should be added below.
    TODO: This "run_model" method signature should be standardized.
    '''
    model = None
    if args.model == "cached":
        model = ServerModelsCachedFormat.CachedModel(args.model_fn)
    elif args.model == "keras":
        if args.model_fn is not None:
            model = ServerModelsICLRFormat.KerasModel(args.model_fn, args.gpuid)
        else:
            print("Must pass --model_fn when using a `keras` model. Exiting...")
            return
    else:
        print("Model isn't implemented, aborting")
        return
    # We pass the dynamically loaded method to the `predPatch` callback as an argument 
    custom_pred_patch = functools.partial(pred_patch, model=model)


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
