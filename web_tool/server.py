#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110
import sys
import os
import time
import datetime
import collections

import bottle

import argparse
import base64
import json
import uuid

import numpy as np
import cv2

import fiona
import fiona.transform

import rasterio
import rasterio.warp

import mercantile

import pickle
import joblib

from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity

from DataLoader import warp_data_to_3857, crop_data_by_extent
from Heatmap import Heatmap
from Datasets import DATASETS
from Utils import get_random_string, class_prediction_to_img, get_shape_layer_by_name, AtomicCounter

from ServerModelsKerasDense import KerasDenseFineTune

from web_tool import ROOT_DIR

import requests
from beaker.middleware import SessionMiddleware
import login
import login_config
from log import Log

import tornado
from tornado.wsgi import WSGIContainer
from tornado.ioloop import IOLoop
from tornado.web import FallbackHandler, RequestHandler, Application
from tornado.httpserver import HTTPServer


class Session():
    ''' Currently this is a totally static class, however this is what needs to change if we are to support multiple sessions.
    '''
    storage_type = None # this will be "table" or "file"
    storage_path = None # this will be a file path
    table_service = None # this will be an instance of TableService

    gpuid = None

    model = None
    current_transform = ()

    current_snapshot_string = get_random_string(8)
    current_snapshot_idx = 0
    current_request_counter = AtomicCounter()
    request_list = []

    @staticmethod
    def reset(soft=False, from_cached=None):
        if not soft:
            Session.model.reset() # can't fail, so don't worry about it
        Session.current_snapshot_string = get_random_string(8)
        Session.current_snapshot_idx = 0
        Session.current_request_counter = AtomicCounter()
        Session.request_list = []

        if Session.storage_type == "table":
            Session.table_service.insert_entity("webtoolsessions",
            {
                "PartitionKey": str(np.random.randint(0,8)),
                "RowKey": str(uuid.uuid4()),
                "session_id": Session.current_snapshot_string,
                "server_hostname": os.uname()[1],
                "server_sys_argv": ' '.join(sys.argv),
                "base_model": from_cached
            })

    @staticmethod
    def load(encoded_model_fn):
        model_fn = base64.b64decode(encoded_model_fn).decode('utf-8')

        print(model_fn)

        del Session.model
        Session.model = joblib.load(model_fn)

    @staticmethod
    def save(model_name):

        if Session.storage_type is not None:
            assert Session.storage_path is not None # we check for this when starting the program

            snapshot_id = "%s_%d" % (model_name, Session.current_snapshot_idx)
            
            print("Saving state for %s" % (snapshot_id))
            base_dir = os.path.join(Session.storage_path, Session.current_snapshot_string)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=False)
            
            model_fn = os.path.join(base_dir, "%s_model.p" % (snapshot_id))
            joblib.dump(Session.model, model_fn, protocol=pickle.HIGHEST_PROTOCOL)

            if Session.storage_type == "file":
                request_list_fn = os.path.join(base_dir, "%s_request_list.p" % (snapshot_id))
                joblib.dump(Session.request_list, request_list_fn, protocol=pickle.HIGHEST_PROTOCOL)
            elif Session.storage_type == "table":
                # We don't serialize the request list when saving to table storage
                pass

            Session.current_snapshot_idx += 1
            return base64.b64encode(model_fn.encode('utf-8')).decode('utf-8') # this is super dumb
        else:
            return None
    
    @staticmethod
    def add_entry(data):
        client_ip = bottle.request.environ.get('HTTP_X_FORWARDED_FOR') or bottle.request.environ.get('REMOTE_ADDR')
        data = data.copy()
        data["time"] = datetime.datetime.now()
        data["remote_address"] = client_ip
        data["current_snapshot_index"] = Session.current_snapshot_idx
        current_request_counter = Session.current_request_counter.increment()
        data["current_request_index"] = current_request_counter

        assert "experiment" in data

        if Session.storage_type == "file":
            Session.request_list.append(data)
        
        elif Session.storage_type == "table":

            data["PartitionKey"] = Session.current_snapshot_string
            data["RowKey"] = "%s_%d" % (data["experiment"], current_request_counter)

            for k in data.keys():
                if isinstance(data[k], dict) or isinstance(data[k], list):
                    data[k] = json.dumps(data[k])
            
            try:
                Session.table_service.insert_entity("webtoolinteractions", data)
            except Exception as e:
                print(e)
        else:
            # The storage_type / --storage_path command line args were not set
            pass

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

def do_heatmap(z,y,x):
    bottle.response.content_type = 'image/jpeg'
    x = x.split("?")[0]
    return Heatmap.get(int(z),int(y),int(x))

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def do_load():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    cached_model = data["cachedModel"]

    Session.reset(False, from_cached=cached_model)
    Session.load(cached_model)

    data["message"] = "Loaded new model from %s" % (cached_model)
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)

def reset_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    initial_reset = data.get("initialReset", False)
    if not initial_reset:
        Session.add_entry(data) # record this interaction
        Session.save(data["experiment"])

    Heatmap.reset()
    Session.reset()

    data["message"] = "Reset model"
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)

def retrain_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    
    success, message = Session.model.retrain(**data["retrainArgs"])
    
    if success:
        bottle.response.status = 200
        encoded_model_fn = Session.save(data["experiment"])
        data["cached_model"] = encoded_model_fn 
        Session.add_entry(data) # record this interaction
    else:
        data["error"] = message
        bottle.response.status = 500

    data["message"] = message
    data["success"] = success

    return json.dumps(data)

def record_correction():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    Session.add_entry(data) # record this interaction

    #
    tlat, tlon = data["extent"]["ymax"], data["extent"]["xmin"]
    blat, blon = data["extent"]["ymin"], data["extent"]["xmax"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    color_list = [item["color"] for item in class_list]
    class_idx = data["value"] # what we want to switch the class to
    origin_crs = "epsg:%d" % (data["extent"]["spatialReference"]["latestWkid"])

    # Add a click to the heatmap
    xs, ys = fiona.transform.transform(origin_crs, "epsg:4326", [tlon], [tlat])
    tile = mercantile.tile(xs[0], -ys[0], 17)
    Heatmap.increment(tile.z, tile.y, tile.x)

    #
    naip_crs, naip_transform, naip_index = Session.current_transform

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

    Session.model.add_sample(tdst_row, bdst_row, tdst_col, bdst_col, class_idx)
    num_corrected = (bdst_row-tdst_row) * (bdst_col-tdst_col)

    data["message"] = "Successfully submitted correction"
    data["success"] = True
    data["count"] = num_corrected

    bottle.response.status = 200
    return json.dumps(data)

def do_undo():
    ''' Method called for POST `/doUndo`
    '''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    Session.add_entry(data) # record this interaction

    # Forward the undo command to the backend model
    success, message, num_undone = Session.model.undo()
    data["message"] = message
    data["success"] = success
    data["count"] = num_undone

    bottle.response.status = 200
    return json.dumps(data)

def pred_patch():
    ''' Method called for POST `/predPatch`'''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    Session.add_entry(data) # record this interaction

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
    Session.current_transform = (naip_crs, naip_transform, naip_index)

    # ------------------------------------------------------
    # Step 3
    #   Run a model on the input data
    #   Apply reweighting
    #   Fix padding
    # ------------------------------------------------------
    output = Session.model.run(naip_data, extent, False)
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


def pred_tile():
    ''' Method called for POST `/predPatch`'''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    Session.add_entry(data) # record this interaction

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


    output = Session.model.run(naip_data, extent, True)
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


def get_input():
    ''' Method called for POST `/getInput`
    '''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    Session.add_entry(data) # record this interaction

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

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def get_root_app():
    if 'logged_in' in bottle.request.session:
        print("in session")
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
    parser = argparse.ArgumentParser(description="Backend Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)

    parser.add_argument(
        '--storage_type',
        action="store", dest="storage_type", type=str,
        choices=["table", "file"],
        default=None
    )
    parser.add_argument("--storage_path", action="store", dest="storage_path", type=str, help="Path to directory where output will be stored", default=None)

    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4444)
    parser.add_argument("--model", action="store", dest="model",
        choices=[
            "keras_dense"
        ],
        help="Model to use", required=True
    )
    parser.add_argument("--fine_tune_seed_data_fn", action="store", dest="fine_tune_seed_data_fn", type=str, help="Path to npz containing seed data to use", default=None)
    parser.add_argument("--fine_tune_layer", action="store", dest="fine_tune_layer", type=int, help="Layer of model to fine tune", default=-2)


    parser.add_argument("--model_fn", action="store", dest="model_fn", type=str, help="Model fn to use", default=None)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", default=None)

    args = parser.parse_args(sys.argv[1:])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpuid is None else str(args.gpuid)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    model = None
    if args.model == "keras_dense":
        model = KerasDenseFineTune(args.model_fn, args.gpuid, args.fine_tune_layer, args.fine_tune_seed_data_fn)
    else:
        raise NotImplementedError("The given model type is not implemented yet.")

    if args.storage_type == "file":
        assert args.storage_path is not None, "You must specify a storage path if you select the 'path' storage type"
        Session.storage_path = args.storage_path
    elif args.storage_type == "table":
        assert args.storage_path is not None, "You must specify a storage path if you select the 'table' storage type"
        Session.storage_path = args.storage_path

        assert "AZURE_ACCOUNT_NAME" in os.environ
        assert "AZURE_ACCOUNT_KEY" in os.environ

        Session.table_service = TableService(
            account_name=os.environ['AZURE_ACCOUNT_NAME'],
            account_key=os.environ['AZURE_ACCOUNT_KEY']
        )
    elif args.storage_type is None:
        assert args.storage_path is None, "You cannot specify a storage path if you do not select a storage type"

    Session.model = model
    Session.storage_type = args.storage_type


    login.manage_session_folders()
    session_opts = {
        'session.type': 'file',
        'session.cookie_expires': 3000,
        'session.data_dir': login.session_folder,
        'session.auto': True
    }
    bottle.TEMPLATE_PATH.insert(0, "./" + ROOT_DIR + "/views")



    # Setup the bottle server 
    app = bottle.Bottle()

    app.add_hook("after_request", enable_cors)
    app.add_hook("before_request", login.setup_request)

    # Login paths
    app.route("/authorized", method="GET", callback=login.load_authorized)
    app.route("/error", method="GET", callback=login.load_error)
    app.route("/notauthorized", method="GET", callback=login.not_authorized)
    app.route("/login", method="GET", callback=login.do_login)
    app.route("/login", method="POST", callback=login.do_login)
    app.route("/logout", method="GET", callback=login.do_logout)
    app.route("/checkAccess", method="POST", callback=login.get_accesstoken)

    # API paths
    app.route("/predPatch", method="OPTIONS", callback=do_options)
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

    app.route("/heatmap/<z>/<y>/<x>", method="GET", callback=do_heatmap)

    # Content paths
    app.route("/", method="GET", callback=get_root_app)
    app.route("/js/datasets.js", method="GET", callback=get_datasets)
    app.route("/favicon.ico", method="GET", callback=get_favicon)
    app.route("/<filepath:re:.*>", method="GET", callback=get_everything_else)

    app = SessionMiddleware(app, session_opts)

    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": args.verbose,
        "server": "tornado",
        "reloader": False,
        "options": {"threads": 12} # TODO: As of bottle version 0.12.17, the WaitressBackend does not get the **options kwargs
    }
    #app.run(**bottle_server_kwargs)
    
    tr = WSGIContainer(app)

    application = Application([
            #(r"/tornado", MainHandler),
            (r".*", FallbackHandler, dict(fallback=tr)),
    ])
        
    http_server = HTTPServer(application, ssl_options={'certfile':login_config.CERT_FILE, 'keyfile': login_config.KEY_FILE})

    http_server.listen(args.port, address=args.host)
    IOLoop.instance().start()

    #from waitress import serve
    #serve(app, host=args.host, port=args.port, threads=12)


if __name__ == "__main__":
    main()
