# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# # /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those
# libraries directly.
from task_management.api_task import ApiTaskManager
from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from time import sleep
import json
from ai4e_app_insights import AppInsights
from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import AI4EWrapper
import sys
import os
from os import getenv

import sys
import os
import functools
import base64
import json

import numpy as np
import cv2

import DataLoader
import GeoTools
import utils
import ServerModelsCached


print("Creating Application")

api_prefix = getenv('API_PREFIX')
app = Flask(__name__)
api = Api(app)


app.config['JSON_SORT_KEYS'] = False

# Log requests, traces and exceptions to the Application Insights service
appinsights = AppInsights(app)

# Use the AI4EAppInsights library to send log messages.
log = AI4EAppInsights()

# Use the internal-container AI for Earth Task Manager (not for production use!).
api_task_manager = ApiTaskManager(flask_api=api, resource_prefix=api_prefix)

# Use the AI4EWrapper to executes your functions within a logging trace.
# Also, helps support long-running/async functions.
ai4e_wrapper = AI4EWrapper(app)

#load the precomputed results
model = ServerModelsCached.run

@app.after_request
def enable_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'PUT, GET, POST, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

    return response
 
@app.route('/', methods=['GET'])
def health_check():
    return "Health check OK"

@app.route(api_prefix + '/predPatch', methods=['POST'])
def post_pred_patch():    
    # wrap_sync_endpoint wraps your function within a logging trace.
    post_data  = request.get_json()
    return ai4e_wrapper.wrap_sync_endpoint(pred_patch, "post:pred_patch", data=post_data)

@app.route(api_prefix + '/getInput', methods=['POST'])
def post_get_input():
    post_data = json.loads(request.data)
    #return(get_input(post_data))
    return ai4e_wrapper.wrap_sync_endpoint(get_input, "post:get_input", data=post_data)

#def pred_patch(**kwargs):
def pred_patch(data):
    # Inputs
  
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
        print(naip_fn)
    except ValueError as e:
        print(e)
        return json.dumps({"error": str(e)})

    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------

    naip_data, padding = DataLoader.get_data_by_extent(naip_fn, extent, DataLoader.GeoDataTypes.NAIP)
    naip_data = np.rollaxis(naip_data, 0, 3)
    
    # ------------------------------------------------------
    # Step 3
    #   Run a model on the input data
    #   Apply reweighting
    #   Fix padding
    # ------------------------------------------------------
    #output, name = ServerModels_Baseline_Blg_test.run_cnn(naip_data, landsat_data, blg_data, with_smooth=False)
    #name += "_with_smooth_False"
    output, name = model(naip_data, naip_fn, extent, padding)

    assert output.shape[2] == 4, "The model function should return an image shaped as (height, width, num_classes)"
    output *= weights[np.newaxis, np.newaxis, :] # multiply by the weight vector
    sum_vals = output.sum(axis=2) # need to normalize sums to 1 in order for the rendered output to be correct
    output = output / (sum_vals[:,:,np.newaxis] + 0.000001)
    
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

    return json.dumps(data)

def get_input(data):
    
    ''' Method called for POST `/getInput`
    '''
    # Inputs
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

    return json.dumps(data)

if __name__ == '__main__':
    app.run()

