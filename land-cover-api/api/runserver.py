# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# # /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those
# libraries directly.
from task_management.api_task import ApiTaskManager
from flask import Flask, request, abort, make_response, jsonify
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from time import sleep
from ai4e_app_insights import AppInsights
from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import AI4EWrapper
from input_validation import *
from os import getenv
from enum import Enum
from datetime import datetime

import sys
import os
import json
import base64
import functools
import numpy as np
import cv2
import DataLoader
import GeoTools
import utils
import ServerModelsCached

import time

print("Creating Application")

api_prefix = getenv('API_PREFIX')
app = Flask(__name__)
api = Api(app)


#-re-enable for log
# Log requests, traces and exceptions to the Application Insights service
appinsights = AppInsights(app)

# # Use the AI4EAppInsights library to send log messages.
log = AI4EAppInsights()

# # Use the internal-container AI for Earth Task Manager (not for production use!).
api_task_manager = ApiTaskManager(flask_api=api, resource_prefix=api_prefix)

# # Use the AI4EWrapper to executes your functions within a logging trace.
# # Also, helps support long-running/async functions.
ai4e_wrapper = AI4EWrapper(app)

#load the precomputed results
model = ServerModelsCached.run

validator = InputValidator()

@app.after_request
def enable_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    #response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

    return response

def abort_error(error_code, error_message):
    return make_response(jsonify({'error': error_message}), error_code)

def get_input_data():
    data = json.loads(request.data)
    #convert all json to lowercase
    data = eval(repr(data).lower())

    return data

@app.route(api_prefix + '/', methods=['GET'])
def health_check():
    return "Health check OK"

@app.route(api_prefix + '/classify', methods=['POST'])
def classify():
    is_valid, msg = validator.validate_input_data(request.data, InputType.latlon,
                                                  RequestType.classify)   
    if not is_valid:
        return abort_error(400, msg)
    
    post_data = get_input_data()

    return ai4e_wrapper.wrap_sync_endpoint(pred_patch, "post:pred_patch", 
                                           data=post_data, 
                                           type=InputType.latlon)
        
@app.route(api_prefix + '/tile', methods=['POST'])
def get_tile():  
    is_valid, msg = validator.validate_input_data(request.data, InputType.latlon,
                                                  RequestType.tile)   
    if not is_valid:
        return abort_error(400, msg)
    
    post_data = get_input_data()

    return ai4e_wrapper.wrap_sync_endpoint(get_input, "post:get_input", 
                                           data=post_data, 
                                           type=InputType.latlon)

@app.route(api_prefix + '/classify_by_extent', methods=['POST'])
def classify_extent():
    is_valid, msg = validator.validate_input_data(request.data, InputType.extent,
                                                  RequestType.classify)   
    if not is_valid:
        return abort_error(400, msg)
    
    post_data = get_input_data()

    return ai4e_wrapper.wrap_sync_endpoint(pred_patch, "post:pred_patch",
                                           data=post_data, 
                                           type=InputType.extent)

@app.route(api_prefix + '/tile_by_extent', methods=['POST'])
def get_tile_by_extent():
    is_valid, msg = validator.validate_input_data(request.data, InputType.extent,
                                                  RequestType.tile)   
    if not is_valid:
        return abort_error(400, msg)
    
    post_data = get_input_data()

    return ai4e_wrapper.wrap_sync_endpoint(get_input, "post:get_input", 
                                           data=post_data, 
                                           type=InputType.extent)

def pred_patch(data, type):
    ''' Method called for POST `/predPatchLatLon` and  POST `/predPatch`
    
    '''
    weights = np.array(data["weights"], dtype=np.float32)
    
    # ------------------------------------------------------
    # Step 1
    #   Transform the input extent into a shapely geometry
    #   Find the tile assosciated with the geometry
    # ------------------------------------------------------
    if(type == InputType.extent):
        
        extent = data["extent"]
        GeoTools.latest_wkid = extent["spatialreference"]["latestwkid"]
        
        geom = GeoTools.extent_to_transformed_geom(extent, "EPSG:4269")
    else:   
        lat = data["lat"]
        lon = data["lon"]
        
        GeoTools.latest_wkid = data["latestwkid"]
        GeoTools.patch_size = data["patchsize"]
        
        extent, geom = GeoTools.get_geom(lat, lon, "EPSG:4326")

        data["extent"] = extent

    try:
        print('runserver, pred_patch, lookup tile by geom:')
        start = datetime.now()        
        
        naip_fn = DataLoader.lookup_tile_by_geom(geom)
        
        stop = datetime.now()
        duration = start - stop
        print('runserver, pred_patch, lookup tile by geom: {} seconds.'.format(duration))
    except ValueError as e:
        error_msg = 'Error occurred in tile retrieval, no data is available for the specified location'
        log.log_exception(error_msg + '(in pred_patch function) :' + str(e))
        return abort_error(400, error_msg)

    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------
    print('runserver, pred_patch, load input data sources: ')
    start = datetime.now()  
    
    naip_data, padding = DataLoader.get_data_by_extent(naip_fn, extent, DataLoader.GeoDataTypes.NAIP)
    naip_data = np.rollaxis(naip_data, 0, 3)
    
    stop = datetime.now()
    duration = start - stop
    print('runserver, pred_patch, load input data sources: {} seconds.'.format(duration))


    # ------------------------------------------------------
    # Step 3
    #   Run a model on the input data
    #   Apply reweighting
    #   Fix padding
    # ------------------------------------------------------
    print('runserver, pred_patch, get cached data for input: ') 
    start = datetime.now()  
    
    output, name = model(naip_data, naip_fn, extent, padding)
    
    assert output.shape[2] == 4, "The model function should return an image shaped as (height, width, num_classes)"
    output *= weights[np.newaxis, np.newaxis, :] # multiply by the weight vector
    sum_vals = output.sum(axis=2) # need to normalize sums to 1 in order for the rendered output to be correct
    output = output / (sum_vals[:,:,np.newaxis] + 0.000001)
    
    stop = datetime.now()
    duration = start - stop
    print('runserver, pred_patch, get cached data for input: {} seconds.'.format(duration))

    # ------------------------------------------------------
    # Step 4
    #   Convert images to base64 and return  
    # ------------------------------------------------------

    print('runserver, pred_patch, return classification images: ')
    start = datetime.now()  

    img_soft = np.round(utils.class_prediction_to_img(output, False)*255,0).astype(np.uint8)
    img_soft = cv2.imencode(".png", cv2.cvtColor(img_soft, cv2.COLOR_RGB2BGR))[1].tostring()
    img_soft = base64.b64encode(img_soft).decode("utf-8")
    data["output_soft"] = img_soft

    img_hard = np.round(utils.class_prediction_to_img(output, True)*255,0).astype(np.uint8)
    img_hard = cv2.imencode(".png", cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGR))[1].tostring()
    img_hard = base64.b64encode(img_hard).decode("utf-8")
    data["output_hard"] = img_hard

    data["model_name"] = name

    stop = datetime.now()
    duration = start - stop
    print('runserver, pred_patch, return classification images: {} seconds.'.format(duration))

    return json.dumps(data)

def get_input(data, type):
    
    ''' Method called for  POST `/getInput` and POST `/getInputLatLon`
    '''

    # ------------------------------------------------------
    # Step 1
    #   Transform the input extent into a shapely geometry
    #   Find the tile assosciated with the geometry
    # ------------------------------------------------------
    if(type == InputType.extent):      
        extent = data["extent"]
        GeoTools.latest_wkid = extent["spatialreference"]["latestwkid"]
        geom = GeoTools.extent_to_transformed_geom(extent, "EPSG:4269")
    else:
        lat = data["lat"]
        lon = data["lon"]
        
        GeoTools.latest_wkid = data["latestwkid"]
        GeoTools.patch_size = data["patchsize"]
        
        extent, geom = GeoTools.get_geom(lat, lon, "EPSG:4326")
        data["extent"] = extent
    try:
        print('runserver, get_input, lookupvtile by geom: ')
        start = datetime.now()    
        
        naip_fn = DataLoader.lookup_tile_by_geom(geom)
        
        stop = datetime.now()
        duration = stop - start
        print('runserver, get_input, lookup tile by geom: {} seconds.'.format(duration))
    except ValueError as e:
        error_msg = 'Error occurred in tile retrieval, no data is available for the specified location'
        log.log_exception(error_msg + '(in get_input function) :' + str(e))
        return abort_error(400, error_msg)

    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------
    print('runserver, get_input, Load the input data sources for the given tile: ')
    start = datetime.now()   

    naip_data, padding = DataLoader.get_data_by_extent(naip_fn, extent, DataLoader.GeoDataTypes.NAIP)
    naip_data = np.rollaxis(naip_data, 0, 3)
    naip_img = naip_data[:,:,:3].copy().astype(np.uint8) # keep the RGB channels to save as a color image later
    if padding > 0:
        naip_img = naip_img[padding:-padding,padding:-padding,:]

    img_naip = cv2.imencode(".png", cv2.cvtColor(naip_img, cv2.COLOR_RGB2BGR))[1].tostring()
    img_naip = base64.b64encode(img_naip).decode("utf-8")
    data["input_naip"] = img_naip

    stop = datetime.now()
    duration = stop - start
    print('runserver, get_input, Load the input data sources for the given tile: {} seconds.'.format(duration))

    return json.dumps(data)

if __name__ == '__main__':
    app.run()

