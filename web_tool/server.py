#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110
import sys
import os
import time
import datetime
import collections
import argparse
import base64
import json
import uuid
import threading

import numpy as np
import cv2

import fiona
import fiona.transform

import rasterio
import rasterio.warp

import pickle
import joblib

from DataLoader import warp_data_to_3857, crop_data_by_extent

from Datasets import DATASETS
from Utils import get_random_string, class_prediction_to_img, get_shape_layer_by_name, AtomicCounter

from web_tool import ROOT_DIR

import bottle 
bottle.TEMPLATE_PATH.insert(0, "./" + ROOT_DIR + "/views") # let bottle know where we are storing the template files

import requests
from beaker.middleware import SessionMiddleware

import login
import login_config
from log import setup_logging, LOGGER

#---------------------------------------------------------------------------------------
# Session handling code
#---------------------------------------------------------------------------------------

from Session import Session
SESSION_MAP = dict() # keys are bottle.request.session.id and values are corresponding Session() objects
EXPIRED_SESSION_SET = set()

def setup_sessions():
    '''Adds the beaker SessionMiddleware on as request.session
    '''
    bottle.request.session = bottle.request.environ['beaker.session']
    bottle.request.client_ip = bottle.request.environ.get('HTTP_X_FORWARDED_FOR') or bottle.request.environ.get('REMOTE_ADDR')

def manage_sessions():

    if bottle.request.session.id in EXPIRED_SESSION_SET:
        bottle.request.session.delete()
        EXPIRED_SESSION_SET.remove(bottle.request.session.id)
    
    if "logged_in" in bottle.request.session:
        if bottle.request.session.id in SESSION_MAP:
            LOGGER.info("Updating last_interaction_time")
            SESSION_MAP[bottle.request.session.id].last_interaction_time = time.time()
        else:
            LOGGER.warning("I don't expect this to happen - someone is logged in but they do not have a `Session` object in SESSION_MAP")
    else:
        pass # not logged in, we don't care about this

def session_monitor():
    ''' This is a `Thread()` that is starting when the program is run. It is responsible for finding which of the `Session()` objects
    in `SESSION_MAP` haven't been used recently and killing them.
    '''
    LOGGER.info("Starting session monitor thread")
    while True:
        session_ids_to_delete = []
        for session_id, session in SESSION_MAP.items():
            LOGGER.info("SESSION MONITOR - Checking session (%s) for activity" % (session_id))
            time_inactive = time.time() - session.last_interaction_time
            if time_inactive > 30:
                LOGGER.info("SESSION MONITOR - Session (%s) has been inactive for over 30 seconds" % (session_id))
                EXPIRED_SESSION_SET.add(session_id)
                session_ids_to_delete.append(session_id) 
        for session_id in session_ids_to_delete:
            del SESSION_MAP[session_id]
        time.sleep(5)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

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
#---------------------------------------------------------------------------------------

@login.authenticated
def do_load():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    cached_model = data["cachedModel"]

    SESSION_MAP[bottle.request.session.id].reset(False, from_cached=cached_model)
    SESSION_MAP[bottle.request.session.id].load(cached_model)

    data["message"] = "Loaded new model from %s" % (cached_model)
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)


@login.authenticated
def reset_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip
    
    initial_reset = data.get("initialReset", False)
    if not initial_reset:
        SESSION_MAP[bottle.request.session.id].add_entry(data) # record this interaction
        SESSION_MAP[bottle.request.session.id].save(data["experiment"])

    SESSION_MAP[bottle.request.session.id].reset()

    data["message"] = "Reset model"
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)


@login.authenticated
def retrain_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip
    
    success, message = SESSION_MAP[bottle.request.session.id].model.retrain(**data["retrainArgs"])
    
    if success:
        bottle.response.status = 200
        encoded_model_fn = SESSION_MAP[bottle.request.session.id].save(data["experiment"])
        data["cached_model"] = encoded_model_fn 
        SESSION_MAP[bottle.request.session.id].add_entry(data) # record this interaction
    else:
        data["error"] = message
        bottle.response.status = 500

    data["message"] = message
    data["success"] = success

    return json.dumps(data)


@login.authenticated
def record_correction():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip

    SESSION_MAP[bottle.request.session.id].add_entry(data) # record this interaction

    #
    tlat, tlon = data["extent"]["ymax"], data["extent"]["xmin"]
    blat, blon = data["extent"]["ymin"], data["extent"]["xmax"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    color_list = [item["color"] for item in class_list]
    class_idx = data["value"] # what we want to switch the class to
    origin_crs = "epsg:%d" % (data["extent"]["spatialReference"]["latestWkid"])

    # record points in lat/lon
    xs, ys = fiona.transform.transform(origin_crs, "epsg:4326", [tlon], [tlat])

    #
    naip_crs, naip_transform, naip_index = SESSION_MAP[bottle.request.session.id].current_transform

    xs, ys = fiona.transform.transform(origin_crs, naip_crs.to_dict(), [tlon,blon], [tlat,blat])
    
    tdst_x = xs[0]
    tdst_y = ys[0]
    tdst_col, tdst_row = (~naip_transform) * (tdst_x, tdst_y)
    tdst_row = int(np.floor(tdst_row))
    tdst_col = int(np.floor(tdst_col))

    bdst_x = xs[1]
    bdst_y = ys[1]
    bdst_col, bdst_row = (~naip_transform) * (bdst_x, bdst_y)
    bdst_row = int(np.floor(bdst_row))
    bdst_col = int(np.floor(bdst_col))

    tdst_row, bdst_row = min(tdst_row, bdst_row), max(tdst_row, bdst_row)
    tdst_col, bdst_col = min(tdst_col, bdst_col), max(tdst_col, bdst_col)

    SESSION_MAP[bottle.request.session.id].model.add_sample(tdst_row, bdst_row, tdst_col, bdst_col, class_idx)
    num_corrected = (bdst_row-tdst_row) * (bdst_col-tdst_col)

    data["message"] = "Successfully submitted correction"
    data["success"] = True
    data["count"] = num_corrected

    bottle.response.status = 200
    return json.dumps(data)


@login.authenticated
def do_undo():
    ''' Method called for POST `/doUndo`
    '''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip

    SESSION_MAP[bottle.request.session.id].add_entry(data) # record this interaction

    # Forward the undo command to the backend model
    success, message, num_undone = SESSION_MAP[bottle.request.session.id].model.undo()
    data["message"] = message
    data["success"] = success
    data["count"] = num_undone

    bottle.response.status = 200
    return json.dumps(data)


@login.authenticated
def pred_patch():
    ''' Method called for POST `/predPatch`'''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip

    SESSION_MAP[bottle.request.session.id].add_entry(data) # record this interaction

    # Inputs
    extent = data["extent"]
    dataset = data["dataset"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    color_list = [item["color"] for item in class_list]

    # ------------------------------------------------------
    # Step 1
    #   Transform the input extent into a shapely geometry
    #   Find the tile assosciated with the geometry
    # ------------------------------------------------------
    
    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------

    if dataset not in DATASETS:
        raise ValueError("Dataset doesn't seem to be valid, do the datasets in js/tile_layers.js correspond to those in TileLayers.py")

    naip_data, naip_crs, naip_transform, naip_bounds, naip_index = DATASETS[dataset]["data_loader"].get_data_from_extent(extent)
    naip_data = np.rollaxis(naip_data, 0, 3) # we do this here instead of get_data_by_extent because not all GeoDataTypes will have a channel dimension
    SESSION_MAP[bottle.request.session.id].current_transform = (naip_crs, naip_transform, naip_index)

    # ------------------------------------------------------
    # Step 3
    #   Run a model on the input data
    #   Apply reweighting
    #   Fix padding
    # ------------------------------------------------------
    output = SESSION_MAP[bottle.request.session.id].model.run(naip_data, extent, False)
    assert len(output.shape) == 3, "The model function should return an image shaped as (height, width, num_classes)"
    assert (output.shape[2] < output.shape[0] and output.shape[2] < output.shape[1]), "The model function should return an image shaped as (height, width, num_classes)" # assume that num channels is less than img dimensions

    # ------------------------------------------------------
    # Step 4
    #   Warp output to EPSG:3857 and crop off the padded area
    # ------------------------------------------------------
    output, output_bounds = warp_data_to_3857(output, naip_crs, naip_transform, naip_bounds)
    output = crop_data_by_extent(output, output_bounds, extent)

    # ------------------------------------------------------
    # Step 5
    #   Convert images to base64 and return  
    # ------------------------------------------------------
    img_soft = np.round(class_prediction_to_img(output, False, color_list)*255,0).astype(np.uint8)
    img_soft = cv2.imencode(".png", cv2.cvtColor(img_soft, cv2.COLOR_RGB2BGR))[1].tostring()
    img_soft = base64.b64encode(img_soft).decode("utf-8")
    data["output_soft"] = img_soft

    img_hard = np.round(class_prediction_to_img(output, True, color_list)*255,0).astype(np.uint8)
    img_hard = cv2.imencode(".png", cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGR))[1].tostring()
    img_hard = base64.b64encode(img_hard).decode("utf-8")
    data["output_hard"] = img_hard

    bottle.response.status = 200
    return json.dumps(data)


@login.authenticated
def pred_tile():
    ''' Method called for POST `/predPatch`'''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip

    SESSION_MAP[bottle.request.session.id].add_entry(data) # record this interaction

    # Inputs
    extent = data["extent"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    color_list = [item["color"] for item in class_list]
    dataset = data["dataset"]
    zone_layer_name = data["zoneLayerName"]
   
    if dataset not in DATASETS:
        raise ValueError("Dataset doesn't seem to be valid, do the datasets in js/tile_layers.js correspond to those in TileLayers.py")    
    
    try:
        naip_data, raster_profile, raster_transform, raster_bounds, raster_crs = DATASETS[dataset]["data_loader"].get_data_from_shape_by_extent(extent, zone_layer_name)
        naip_data = np.rollaxis(naip_data, 0, 3)
        shape_area = DATASETS[dataset]["data_loader"].get_area_from_shape_by_extent(extent, zone_layer_name)      
    except NotImplementedError as e:
        bottle.response.status = 400
        return json.dumps({"error": "Cannot currently download imagery with 'Basemap' based datasets"})


    output = SESSION_MAP[bottle.request.session.id].model.run(naip_data, extent, True)
    output_hard = output.argmax(axis=2)
    print("Finished, output dimensions:", output.shape)

    # apply nodata mask from naip_data
    nodata_mask = np.sum(naip_data == 0, axis=2) == 4
    output_hard[nodata_mask] = 255
    vals, counts = np.unique(output_hard[~nodata_mask], return_counts=True)

    # ------------------------------------------------------
    # Step 4
    #   Convert images to base64 and return  
    # ------------------------------------------------------
    tmp_id = get_random_string(8)
    img_hard = np.round(class_prediction_to_img(output, True, color_list)*255,0).astype(np.uint8)
    img_hard = cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGRA)
    img_hard[nodata_mask] = [0,0,0,0]

    img_hard, img_hard_bounds = warp_data_to_3857(img_hard, raster_crs, raster_transform, raster_bounds, resolution=10)

    cv2.imwrite(os.path.join(ROOT_DIR, "downloads/%s.png" % (tmp_id)), img_hard)
    data["downloadPNG"] = "downloads/%s.png" % (tmp_id)

    new_profile = raster_profile.copy()
    new_profile['driver'] = 'GTiff'
    new_profile['dtype'] = 'uint8'
    new_profile['compress'] = "lzw"
    new_profile['count'] = 1
    new_profile['transform'] = raster_transform
    new_profile['height'] = naip_data.shape[0] 
    new_profile['width'] = naip_data.shape[1]
    new_profile['nodata'] = 255
    f = rasterio.open(os.path.join(ROOT_DIR, "downloads/%s.tif" % (tmp_id)), 'w', **new_profile)
    f.write(output_hard.astype(np.uint8), 1)
    f.close()
    data["downloadTIFF"] = "downloads/%s.tif" % (tmp_id)

    f = open(os.path.join(ROOT_DIR, "downloads/%s.txt" % (tmp_id)), "w")
    f.write("Class id\tClass name\tPercent area\tArea (km^2)\n")
    for i in range(len(vals)):
        pct_area = (counts[i] / np.sum(counts))
        if shape_area is not None:
            real_area = shape_area * pct_area
        else:
            real_area = -1
        f.write("%d\t%s\t%0.4f%%\t%0.4f\n" % (vals[i], name_list[vals[i]], pct_area*100, real_area))
    f.close()
    data["downloadStatistics"] = "downloads/%s.txt" % (tmp_id)

    bottle.response.status = 200
    return json.dumps(data)


@login.authenticated
def get_input():
    ''' Method called for POST `/getInput`
    '''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["remote_address"] = bottle.request.client_ip

    SESSION_MAP[bottle.request.session.id].add_entry(data) # record this interaction

    # Inputs
    extent = data["extent"]
    dataset = data["dataset"]

    if dataset not in DATASETS:
        raise ValueError("Dataset doesn't seem to be valid, please check Datasets.py")

    naip_data, naip_crs, naip_transform, naip_bounds, naip_index = DATASETS[dataset]["data_loader"].get_data_from_extent(extent)
    naip_data = np.rollaxis(naip_data, 0, 3)

    naip_data, new_bounds = warp_data_to_3857(naip_data, naip_crs, naip_transform, naip_bounds)
    naip_data = crop_data_by_extent(naip_data, new_bounds, extent)

    naip_img = naip_data[:,:,:3].copy().astype(np.uint8) # keep the RGB channels to save as a color image

    naip_img = cv2.imencode(".png", cv2.cvtColor(naip_img, cv2.COLOR_RGB2BGR))[1].tostring()
    naip_img = base64.b64encode(naip_img).decode("utf-8")
    data["input_naip"] = naip_img

    bottle.response.status = 200
    return json.dumps(data)


@login.authenticated
def whoami():
    return str(bottle.request.session) + " " + str(bottle.request.session.id)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def get_root_app():
    if 'logged_in' in bottle.request.session:
        return bottle.static_file("lg_platform.html", root="./" + ROOT_DIR + "/")
    else:
        return bottle.template("landing_page.tpl")

def get_datasets():
    tile_layers = "var tileLayers = {\n"
    for dataset_name, dataset in DATASETS.items():
        tile_layers += '"%s": %s,\n' % (dataset_name, dataset["javascript_string"])
    tile_layers += "};"
    
    interesting_locations = '''var interestingLocations = [
        L.marker([47.60, -122.15]).bindPopup('Bellevue, WA'),
        L.marker([39.74, -104.99]).bindPopup('Denver, CO'),
        L.marker([37.53,  -77.44]).bindPopup('Richmond, VA'),
        L.marker([39.74, -104.99]).bindPopup('Denver, CO'),
        L.marker([37.53,  -77.44]).bindPopup('Richmond, VA'),
        L.marker([33.746526, -84.387522]).bindPopup('Atlanta, GA'),
        L.marker([32.774250, -96.796122]).bindPopup('Dallas, TX'),
        L.marker([40.106675, -88.236409]).bindPopup('Champaign, IL'),
        L.marker([38.679485, -75.874667]).bindPopup('Dorchester County, MD'),
        L.marker([34.020618, -118.464412]).bindPopup('Santa Monica, CA'),
        L.marker([37.748517, -122.429771]).bindPopup('San Fransisco, CA'),
        L.marker([38.601951, -98.329227]).bindPopup('Ellsworth County, KS')
    ];'''

    return tile_layers + '\n\n' + interesting_locations

def get_favicon():
    return

def get_everything_else(filepath):
    return bottle.static_file(filepath, root="./" + ROOT_DIR + "/")

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI for Earth Land Cover")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)

    # TODO: make sure the storage type is passed onto the Session objects
    parser.add_argument(
        '--storage_type',
        action="store", dest="storage_type", type=str,
        choices=["table", "file"],
        default=None
    )
    parser.add_argument("--storage_path", action="store", dest="storage_path", type=str, help="Path to directory where output will be stored", default=None)

    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4444)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", default=None)


    args = parser.parse_args(sys.argv[1:])

    # Setup logging
    log_path = os.getcwd() + "/logs"
    setup_logging(log_path, "server")


    # Setup the bottle server 
    app = bottle.Bottle()

    app.add_hook("after_request", enable_cors)
    app.add_hook("before_request", setup_sessions)
    app.add_hook("before_request", manage_sessions)

    # Login paths
    app.route("/authorized", method="GET", callback=login.load_authorized)
    app.route("/error", method="GET", callback=login.load_error)
    app.route("/notAuthorized", method="GET", callback=login.not_authorized)
    app.route("/login", method="GET", callback=login.do_login)
    app.route("/login", method="POST", callback=login.do_login)
    app.route("/logout", method="GET", callback=login.do_logout)
    app.route("/checkAccess", method="POST", callback=lambda :login.get_accesstoken(SESSION_MAP))

    # API paths
    app.route("/predPatch", method="OPTIONS", callback=do_options)  # TODO: all of our web requests from index.html fire an OPTIONS call because of https://stackoverflow.com/questions/1256593/why-am-i-getting-an-options-request-instead-of-a-get-request, we should fix this 
    app.route('/predPatch', method="POST", callback=pred_patch)

    app.route("/predTile", method="OPTIONS", callback=do_options)
    app.route('/predTile', method="POST", callback=pred_tile)
    
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

    app.route("/doLoad", method="OPTIONS", callback=do_options)
    app.route("/doLoad", method="POST", callback=do_load)

    app.route("/whoami", method="GET", callback=whoami)

    # Content paths
    app.route("/", method="GET", callback=get_root_app)
    app.route("/js/datasets.js", method="GET", callback=get_datasets)
    app.route("/favicon.ico", method="GET", callback=get_favicon)
    app.route("/<filepath:re:.*>", method="GET", callback=get_everything_else)


    login.manage_session_folders()
    session_opts = {
        'session.type': 'file',
        'session.cookie_expires': 3000,
        'session.data_dir': login.session_folder,
        'session.auto': True
    }
    app = SessionMiddleware(app, session_opts)

    session_monitor_thread = threading.Thread(target=session_monitor)
    session_monitor_thread.setDaemon(True)
    session_monitor_thread.start()

    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": args.verbose,
        "server": "cheroot", # we have switched to the "cheroot" backend, this depends on the development bottle build for some reason 
        "reloader": False,
        "certfile": login_config.CERT_FILE,
        "keyfile": login_config.KEY_FILE,
    }
    bottle.run(app, **bottle_server_kwargs)


if __name__ == "__main__":
    main()
