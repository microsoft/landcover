#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110,E1101
import argparse
import base64
import json
import logging
import os
os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt" 

import sys
import time

import cv2
import fiona
import fiona.transform
import numpy as np
import rasterio
import rasterio.warp

LOGGER = logging.getLogger("server")

from web_tool.DataLoader import warp_data_to_3857, crop_data_by_extent, crop_data_by_geometry, get_area_from_geometry
from web_tool.Datasets import load_datasets
DATALOADERS = load_datasets()

from web_tool.Utils import setup_logging, get_random_string, class_prediction_to_img
from web_tool import ROOT_DIR
from web_tool.Session import manage_session_folders, SESSION_FOLDER
from web_tool.SessionHandler import SessionHandler
from web_tool.Checkpoints import Checkpoints
SESSION_HANDLER = None

import bottle 
bottle.TEMPLATE_PATH.insert(0, "./" + ROOT_DIR + "/views") # let bottle know where we are storing the template files
import cheroot.wsgi
import beaker.middleware

SESSION_TIMEOUT_SECONDS = 900


#---------------------------------------------------------------------------------------
# before_request and after_request methods
#---------------------------------------------------------------------------------------

def manage_sessions():
    '''This method is called before every request.
    
    Adds the beaker SessionMiddleware class on as request.session.

    Checks to see if there a session assosciated with the current request.
    If there is then update the last interaction time on that session.
    '''
    bottle.request.session = bottle.request.environ['beaker.session']
    bottle.request.client_ip = bottle.request.environ.get('HTTP_X_FORWARDED_FOR') or bottle.request.environ.get('REMOTE_ADDR')

    if SESSION_HANDLER.is_expired(bottle.request.session.id): # Someone is trying to use a session that we have deleted due to inactivity
        SESSION_HANDLER.cleanup_expired_session(bottle.request.session.id)
        bottle.request.session.delete() # This sets a Set-cookie header to expire the current bottle.request.session.id on the frontend
        LOGGER.info("Cleaned up an out of date session")
    elif not SESSION_HANDLER.is_active(bottle.request.session.id):
        LOGGER.debug("We are getting a request that doesn't have an active session")
    else:
        if bottle.request.path == "/getSessionStatus":
            pass # getting the session status should not trigger a touch_session
        else:
            SESSION_HANDLER.touch_session(bottle.request.session.id) # let the SESSION_HANDLER know that this session has activity


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


#---------------------------------------------------------------------------------------
# Session handling endpoints
#---------------------------------------------------------------------------------------

def create_session():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    SESSION_HANDLER.create_session(bottle.request.session.id, data["dataset"], data["model"], data["checkpoint"])
    
    bottle.response.status = 200
    return json.dumps(data)


def kill_session():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    try:
        SESSION_HANDLER.kill_session(bottle.request.session.id)
        SESSION_HANDLER.cleanup_expired_session(bottle.request.session.id)
    except ValueError as e:
        LOGGER.info(e)

    bottle.request.session.delete()
    bottle.response.status = 200
    return json.dumps(data)

def get_session_status():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    data["sessionID"] = str(bottle.request.session.id)
    data["isActive"] = SESSION_HANDLER.is_active(bottle.request.session.id)

    bottle.response.status = 200
    return json.dumps(data)

def whoami():
    page = f"""
    Your <b>bottle</b> session object: {str(bottle.request.session)} <br/>
    Your <b>bottle</b> session id: {str(bottle.request.session.id)} <br /> <br />

    List of <b>bottle</b> session ids that the server has registered as valid <i>Session</i> objects:
    <ul>
    """

    for session_id, session in SESSION_HANDLER._SESSION_MAP.items():
        page += f"<li>{str(session_id)}</li>"
    page += "</ul>"

    active_session = SESSION_HANDLER.is_active(bottle.request.session.id)
    page += f"<br/><br/>Your session is active: {active_session}"

    return page


#---------------------------------------------------------------------------------------
# API endpoints
#---------------------------------------------------------------------------------------

def reset_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    result = SESSION_HANDLER.get_session(bottle.request.session.id).reset()

    bottle.response.status = 200 if result["success"] else 500
    return json.dumps(result)


def retrain_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    result = SESSION_HANDLER.get_session(bottle.request.session.id).model.retrain(**data["retrainArgs"])
    
    bottle.response.status = 200 if result["success"] else 500
    return json.dumps(result)


def do_undo():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    result = SESSION_HANDLER.get_session(bottle.request.session.id).model.undo()
    
    bottle.response.status = 200 if result["success"] else 500
    return json.dumps(result)


def record_correction():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    current_session = SESSION_HANDLER.get_session(bottle.request.session.id)

    lon, lat = data["point"]["x"], data["point"]["y"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    color_list = [item["color"] for item in class_list]
    class_idx = data["value"] # what we want to switch the class to
    origin_crs = data["point"]["crs"]
    model_idx = data["modelIdx"]

    # load the current predicted patches crs and transform
    data_crs = current_session.latest_input_raster.crs
    data_transform = current_session.latest_input_raster.transform

    x, y = fiona.transform.transform(origin_crs, data_crs, [lon], [lat])
    x = x[0]
    y = y[0]

    dst_col, dst_row = (~data_transform) * (x,y)
    dst_row = int(np.floor(dst_row))
    dst_col = int(np.floor(dst_col))

    result = current_session.model.add_sample_point(dst_row, dst_col, class_idx)

    bottle.response.status = 200 if result["success"] else 500
    return json.dumps(result)


def pred_patch():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    current_session = SESSION_HANDLER.get_session(bottle.request.session.id)

    current_session.add_entry(data) # record this interaction

    # Inputs
    extent = data["extent"]
    dataset = data["dataset"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    color_list = [item["color"] for item in class_list]

    if dataset not in DATALOADERS:
        raise ValueError("Dataset doesn't seem to be valid, do the datasets in js/tile_layers.js correspond to those in TileLayers.py")
    else:
        current_data_loader = DATALOADERS[dataset]

    input_raster = current_data_loader.get_data_from_extent(extent)
    current_session.latest_input_raster = input_raster

    output_raster = current_session.pred_patch(input_raster) # run inference
    warped_output_raster = warp_data_to_3857(output_raster) # warp output to 3857
    cropped_warped_output_raster = crop_data_by_extent(warped_output_raster, extent) # crop to the desired result

    if cropped_warped_output_raster.shape[2] > len(color_list):
       LOGGER.warning("The number of output channels is larger than the given color list, cropping output to number of colors (you probably don't want this to happen")
       cropped_warped_output_raster.data = cropped_warped_output_raster.data[:,:,:len(color_list)]


    # Create color versions of predictions
    img_soft = class_prediction_to_img(cropped_warped_output_raster.data, False, color_list)
    img_soft = cv2.imencode(".png", cv2.cvtColor(img_soft, cv2.COLOR_RGB2BGR))[1].tostring()
    img_soft = base64.b64encode(img_soft).decode("utf-8")
    data["output_soft"] = img_soft

    img_hard = class_prediction_to_img(cropped_warped_output_raster.data, True, color_list)
    img_hard = cv2.imencode(".png", cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGR))[1].tostring()
    img_hard = base64.b64encode(img_hard).decode("utf-8")
    data["output_hard"] = img_hard

    bottle.response.status = 200
    return json.dumps(data)


def pred_tile():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    current_session = SESSION_HANDLER.get_session(bottle.request.session.id)

    current_session.add_entry(data) # record this interaction

    # Inputs
    geom = data["polygon"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    color_list = [item["color"] for item in class_list]
    dataset = data["dataset"]
    zone_layer_name = data["zoneLayerName"]
    model_idx = data["modelIdx"]

    if dataset not in DATALOADERS:
        raise ValueError("Dataset doesn't seem to be valid, do the datasets in js/tile_layers.js correspond to those in TileLayers.py")    
    else:
        current_data_loader = DATALOADERS[dataset]

    try:
        input_raster = current_data_loader.get_data_from_geometry(geom["geometry"])
        shape_area = get_area_from_geometry(geom["geometry"])
    except NotImplementedError as e: # Example of how to handle errors from the rest of the server
        bottle.response.status = 400
        return json.dumps({"error": "Cannot currently download imagery with 'Basemap' based datasets"})
    
    output_raster = current_session.pred_tile(input_raster)
    if output_raster.shape[2] > len(color_list):
       LOGGER.warning("The number of output channels is larger than the given color list, cropping output to number of colors (you probably don't want this to happen")
       output_raster.data = output_raster.data[:,:,:len(color_list)]
    
    output_hard = output_raster.data.argmax(axis=2)
    nodata_mask = np.sum(input_raster.data == 0, axis=2) == input_raster.shape[2]
    output_hard[nodata_mask] = 255
    class_vals, class_counts = np.unique(output_hard[~nodata_mask], return_counts=True)

    img_hard = class_prediction_to_img(output_raster.data, True, color_list)
    img_hard = cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGRA)
    img_hard[nodata_mask] = [0,0,0,0]

    # replace the output predictions with our image data because we are too lazy to make a new InMemoryRaster
    output_raster.data = img_hard
    output_raster.shape = img_hard.shape

    warped_output_raster = warp_data_to_3857(output_raster) # warp output to 3857
    cropped_warped_output_raster = crop_data_by_geometry(warped_output_raster, geom["geometry"], "epsg:4326") # crop to the desired shape
    img_hard = cropped_warped_output_raster.data

    tmp_id = get_random_string(8)
    cv2.imwrite("tmp/downloads/%s.png" % (tmp_id), img_hard)
    data["downloadPNG"] = "tmp/downloads/%s.png" % (tmp_id)

    new_profile = {}
    new_profile['driver'] = 'GTiff'
    new_profile['dtype'] = 'uint8'
    new_profile['compress'] = "lzw"
    new_profile['count'] = 1
    new_profile['transform'] = output_raster.transform
    new_profile['height'] = output_hard.shape[0] 
    new_profile['width'] = output_hard.shape[1]
    new_profile['nodata'] = 255
    with rasterio.open("tmp/downloads/%s.tif" % (tmp_id), 'w', **new_profile) as f:
        f.write(output_hard.astype(np.uint8), 1)
    data["downloadTIFF"] = "tmp/downloads/%s.tif" % (tmp_id)


    data["classStatistics"] = []

    f = open("tmp/downloads/%s.txt" % (tmp_id), "w")
    f.write("Class id\tClass name\tPercent area\tArea (km^2)\n")
    for i in range(len(class_vals)):
        pct_area = (class_counts[i] / np.sum(class_counts))
        if shape_area is not None:
            real_area = shape_area * pct_area
        else:
            real_area = -1
        f.write("%d\t%s\t%0.4f%%\t%0.4f\n" % (class_vals[i], name_list[class_vals[i]], pct_area*100, real_area))
        data["classStatistics"].append({
            "Class ID": int(class_vals[i]),
            "Class Name": name_list[class_vals[i]],
            "Percent Area": float(pct_area),
            "Area (km2)": float(real_area)
        })
    f.close()
    data["downloadStatistics"] = "tmp/downloads/%s.txt" % (tmp_id)

    bottle.response.status = 200
    return json.dumps(data)

def download_all():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    current_session = SESSION_HANDLER.get_session(bottle.request.session.id)

    bottle.response.status = 200
    return json.dumps(data)

def get_input():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    current_session = SESSION_HANDLER.get_session(bottle.request.session.id)
    
    current_session.add_entry(data) # record this interaction

    # Inputs
    extent = data["extent"]
    dataset = data["dataset"]

    if dataset not in DATALOADERS:
        raise ValueError("Dataset doesn't seem to be valid, please check Datasets.py")
    else:
        current_data_loader = DATALOADERS[dataset]

    input_raster = current_data_loader.get_data_from_extent(extent)
    warped_output_raster = warp_data_to_3857(input_raster) # warp image to 3857
    cropped_warped_output_raster = crop_data_by_extent(warped_output_raster, extent) # crop to the desired extent
    
    img = cropped_warped_output_raster.data[:,:,:3].copy().astype(np.uint8) # keep the RGB channels to save as a color image
    img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tostring()
    img = base64.b64encode(img).decode("utf-8")
    data["input_img"] = img

    bottle.response.status = 200
    return json.dumps(data)


#---------------------------------------------------------------------------------------
# Checkpoint handling endpoints
#---------------------------------------------------------------------------------------

def checkpoint_wrapper(disable_checkpoints):
    def create_checkpoint():
        bottle.response.content_type = 'application/json'
        data = bottle.request.json
        
        if disable_checkpoints:
            result = {
                "success": False,
                "message": "Saving checkpoints is disabled on the server"
            }
        else:
            result = SESSION_HANDLER.get_session(bottle.request.session.id).create_checkpoint(data["dataset"], data["model"], data["checkpointName"], data["classes"])

        bottle.response.status = 200 if result["success"] else 500
        return json.dumps(result)
    return create_checkpoint

def get_checkpoints():
    checkpoints = Checkpoints.list_checkpoints()
    return json.dumps(checkpoints, indent=2)

#---------------------------------------------------------------------------------------
# Static file serving endpoints
#---------------------------------------------------------------------------------------

def get_landing_page():
    return bottle.static_file("landing_page.html", root="./" + ROOT_DIR + "/")

def get_basemap_data(filepath):
    return bottle.static_file(filepath, root="./data/basemaps/")

def get_zone_data(filepath):
    return bottle.static_file(filepath, root="./data/zones/")

def get_downloads(filepath):
    return bottle.static_file(filepath, root="./tmp/downloads/")

def get_favicon():
    return

def get_everything_else(filepath):
    return bottle.static_file(filepath, root="./" + ROOT_DIR + "/")


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def main():
    global SESSION_HANDLER
    parser = argparse.ArgumentParser(description="AI for Earth Land Cover")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=8080)

    parser.add_argument("--disable_checkpoints", action="store_true", help="Disables the ability to save checkpoints on the server")


    args = parser.parse_args(sys.argv[1:])

    # Create session factory to handle incoming requests
    SESSION_HANDLER = SessionHandler(args)
    SESSION_HANDLER.start_monitor(SESSION_TIMEOUT_SECONDS)

    # Setup logging
    log_path = os.path.join(os.getcwd(), "tmp/logs/")
    setup_logging(log_path, "server")

    # Make sure some directories exist
    os.makedirs("tmp/checkpoints/", exist_ok=True)
    os.makedirs("tmp/downloads/", exist_ok=True)
    os.makedirs("tmp/logs/", exist_ok=True)
    os.makedirs("tmp/output/", exist_ok=True) # TODO: Remove this after we rework  
    os.makedirs("tmp/session/", exist_ok=True)



    # Setup the bottle server 
    app = bottle.Bottle()

    app.add_hook("after_request", enable_cors)
    app.add_hook("before_request", manage_sessions) # before every request we want to check to make sure there are no session issues

    # API paths
    app.route("/predPatch", method="OPTIONS", callback=do_options) # TODO: all of our web requests from index.html fire an OPTIONS call because of https://stackoverflow.com/questions/1256593/why-am-i-getting-an-options-request-instead-of-a-get-request, we should fix this 
    app.route('/predPatch', method="POST", callback=pred_patch)

    app.route("/predTile", method="OPTIONS", callback=do_options)
    app.route('/predTile', method="POST", callback=pred_tile)

    app.route("/downloadAll", method="OPTIONS", callback=do_options)
    app.route('/downloadAll', method="POST", callback=download_all)
    
    app.route("/getInput", method="OPTIONS", callback=do_options)
    app.route('/getInput', method="POST", callback=get_input)

    app.route("/recordCorrection", method="OPTIONS", callback=do_options)
    app.route('/recordCorrection', method="POST", callback=record_correction)

    app.route("/retrainModel", method="OPTIONS", callback=do_options)
    app.route('/retrainModel', method="POST", callback=retrain_model)

    app.route("/resetModel", method="OPTIONS", callback=do_options)
    app.route('/resetModel', method="POST", callback=reset_model)

    app.route("/doUndo", method="OPTIONS", callback=do_options)
    app.route("/doUndo", method="POST", callback=do_undo)

    app.route("/createSession", method="OPTIONS", callback=do_options)
    app.route("/createSession", method="POST", callback=create_session)

    app.route("/killSession", method="OPTIONS", callback=do_options)
    app.route("/killSession", method="POST", callback=kill_session)

    app.route("/getSessionStatus", method="OPTIONS", callback=do_options)
    app.route("/getSessionStatus", method="POST", callback=get_session_status)

    # Checkpoints
    app.route("/createCheckpoint", method="OPTIONS", callback=do_options)
    app.route("/createCheckpoint", method="POST", callback=checkpoint_wrapper(args.disable_checkpoints))
    app.route("/getCheckpoints", method="GET", callback=get_checkpoints)

    # Sessions
    app.route("/whoami", method="GET", callback=whoami)

    # Content paths
    app.route("/", method="GET", callback=get_landing_page)
    app.route("/data/basemaps/<filepath:re:.*>", method="GET", callback=get_basemap_data)
    app.route("/data/zones/<filepath:re:.*>", method="GET", callback=get_zone_data)
    app.route("/tmp/downloads/<filepath:re:.*>", method="GET", callback=get_downloads)
    app.route("/favicon.ico", method="GET", callback=get_favicon)
    app.route("/<filepath:re:.*>", method="GET", callback=get_everything_else)


    manage_session_folders()
    session_opts = {
        'session.type': 'file',
        #'session.cookie_expires': 3000, # session cookie
        'session.data_dir': SESSION_FOLDER,
        'session.auto': True
    }
    app = beaker.middleware.SessionMiddleware(app, session_opts)

    server = cheroot.wsgi.Server(
        (args.host, args.port),
        app
    )
    server.max_request_header_size = 2**13
    server.max_request_body_size = 2**27

    LOGGER.info("Server initialized")
    try:
        server.start()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
