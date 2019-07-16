#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: disable=E1137,E1136,E0110
import sys
import os
import time
import collections

import bottle
import argparse
import base64
import json

import numpy as np
import cv2

import fiona
import fiona.transform

import rasterio
import rasterio.warp

import mercantile

import DataLoader
import Utils

import pickle
import joblib

import matplotlib
matplotlib.use("Agg") 
import matplotlib.cm

from web_tool.frontend_server import ROOT_DIR

from TileLayers import DataLayerTypes, DATA_LAYERS
import ServerModelsICLRFormat, ServerModelsCachedFormat, ServerModelsICLRDynamicFormat, ServerModelsNIPS, ServerModelsNIPSGroupNorm

def get_random_string(length):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join([alphabet[np.random.randint(0, len(alphabet))] for i in range(length)])

class Heatmap():
    count_dict = collections.defaultdict(int)
    cmap = matplotlib.cm.get_cmap("Reds")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=20, clip=True)

    @staticmethod
    def increment(z,y,x):
        #print("Incrementing", (x,y,z))
        while z > 1:
            key = (z,y,x)
            Heatmap.count_dict[key] += 1
            tile = mercantile.Tile(x,y,z)
            tile = mercantile.parent(tile)
            x,y,z = tile.x, tile.y, tile.z

    def get(z,y,x):
        #print("Getting", (x,y,z))
        key = (z,y,x)
        val = Heatmap.count_dict[key]
        img = np.zeros((256,256,4), dtype=np.uint8)
        if val != 0:
            img[:,:] = np.round(np.array(Heatmap.cmap(Heatmap.norm(val))) * 255).astype(int)

        img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))[1].tostring()
        return img

    def reset():
        Heatmap.count_dict = collections.defaultdict(int)


class AugmentationState():
    #BASE_DIR = "output/"
    debug_mode = False
    BASE_DIR = "/mnt/blobfuse/pred-output/user_study/testing/"
    current_snapshot_string = get_random_string(8)
    current_snapshot_idx = 0
    model = None

    current_transform = ()
    current_naip = None
    current_output = None

    request_list = []

    @staticmethod
    def reset(soft=False):
        if not soft:
            AugmentationState.model.reset() # can't fail, so don't worry about it
        AugmentationState.current_snapshot_string = get_random_string(8)
        if not AugmentationState.debug_mode:
            os.makedirs(os.path.join(AugmentationState.BASE_DIR, AugmentationState.current_snapshot_string))

        AugmentationState.current_snapshot_idx = 0
        AugmentationState.request_list = []

    @staticmethod
    def save(model_name):
        snapshot_id = "%s_%d" % (model_name, AugmentationState.current_snapshot_idx)

        print("Saving state for %s" % (snapshot_id))
        model_fn = os.path.join(AugmentationState.BASE_DIR, AugmentationState.current_snapshot_string, "%s_model.p" % (snapshot_id))
        request_list_fn = os.path.join(AugmentationState.BASE_DIR, AugmentationState.current_snapshot_string, "%s_request_list.p" % (snapshot_id))

        if not AugmentationState.debug_mode:
            joblib.dump(AugmentationState.model, model_fn, protocol=pickle.HIGHEST_PROTOCOL)
            joblib.dump(AugmentationState.request_list, request_list_fn, protocol=pickle.HIGHEST_PROTOCOL)
            AugmentationState.current_snapshot_idx += 1
        
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

def reset_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data) # record this interaction

    Heatmap.reset()

    AugmentationState.save(data["experiment"])
    AugmentationState.reset()

    data["message"] = "Reset model"
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)

def retrain_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data) # record this interaction
    
    #
    success, message = AugmentationState.model.retrain(**data["retrainArgs"])
    if success:
        bottle.response.status = 200
        AugmentationState.save(data["experiment"])
    else:
        data["error"] = message
        bottle.response.status = 500

    data["message"] = message
    data["success"] = success

    return json.dumps(data)

def record_correction():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data) # record this interaction

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
    naip_crs, naip_transform, naip_index, padding = AugmentationState.current_transform

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

    AugmentationState.model.add_sample(tdst_row, bdst_row, tdst_col, bdst_col, class_idx)
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
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data) # record this interaction

    # Forward the undo command to the backend model
    success, message, num_undone = AugmentationState.model.undo()
    data["message"] = message
    data["success"] = success
    data["count"] = num_undone

    bottle.response.status = 200
    return json.dumps(data)

def pred_patch():
    ''' Method called for POST `/predPatch`'''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data) # record this interaction

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

    if dataset not in DATA_LAYERS:
        raise ValueError("Dataset doesn't seem to be valid, do the datasets in js/tile_layers.js correspond to those in TileLayers.py")

    if DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.ESRI_WORLD_IMAGERY:
        padding = 0.0005
        naip_data, naip_crs, naip_transform, naip_bounds, naip_index = DataLoader.get_esri_data_by_extent(extent, padding=padding)
    elif DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.USA_NAIP_LIST:
        padding = 20
        naip_data, naip_crs, naip_transform, naip_bounds, naip_index = DataLoader.get_usa_data_by_extent(extent, padding=padding, geo_data_type=DataLoader.GeoDataTypes.NAIP)
    elif DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.CUSTOM:
        if "padding" in DATA_LAYERS[dataset]:
            padding = DATA_LAYERS[dataset]["padding"]
        else:
            padding = 20
        naip_data, naip_crs, naip_transform, naip_bounds, naip_index = DataLoader.get_custom_data_by_extent(extent, padding=padding, data_fn=DATA_LAYERS[dataset]["data_fn"])

    naip_data = np.rollaxis(naip_data, 0, 3) # we do this here instead of get_data_by_extent because not all GeoDataTypes will have a channel dimension
    AugmentationState.current_transform = (naip_crs, naip_transform, naip_index, padding)

    # ------------------------------------------------------
    # Step 3
    #   Run a model on the input data
    #   Apply reweighting
    #   Fix padding
    # ------------------------------------------------------
    output = AugmentationState.model.run(naip_data, extent, False)
    assert len(output.shape) == 3, "The model function should return an image shaped as (height, width, num_classes)"
    assert (output.shape[2] < output.shape[0] and output.shape[2] < output.shape[1]), "The model function should return an image shaped as (height, width, num_classes)" # assume that num channels is less than img dimensions

    # ------------------------------------------------------
    # Step 4
    #   Warp output to EPSG:3857 and crop off the padded area
    # ------------------------------------------------------
    output, output_bounds = DataLoader.warp_data_to_3857(output, naip_crs, naip_transform, naip_bounds)
    output = DataLoader.crop_data_by_extent(output, output_bounds, extent)

    # ------------------------------------------------------
    # Step 5
    #   Convert images to base64 and return  
    # ------------------------------------------------------
    img_soft = np.round(Utils.class_prediction_to_img(output, False, color_list)*255,0).astype(np.uint8)
    img_soft = cv2.imencode(".png", cv2.cvtColor(img_soft, cv2.COLOR_RGB2BGR))[1].tostring()
    img_soft = base64.b64encode(img_soft).decode("utf-8")
    data["output_soft"] = img_soft

    img_hard = np.round(Utils.class_prediction_to_img(output, True, color_list)*255,0).astype(np.uint8)
    img_hard = cv2.imencode(".png", cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGR))[1].tostring()
    img_hard = base64.b64encode(img_hard).decode("utf-8")
    data["output_hard"] = img_hard

    bottle.response.status = 200
    return json.dumps(data)


def pred_tile():
    ''' Method called for POST `/predPatch`'''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data) # record this interaction

    # Inputs
    extent = data["extent"]
    class_list = data["classes"]
    name_list = [item["name"] for item in class_list]
    color_list = [item["color"] for item in class_list]
    dataset = data["dataset"]
    zone_layer_name = data["zoneLayerName"]
   

    if dataset not in DATA_LAYERS:
        raise ValueError("Dataset doesn't seem to be valid, do the datasets in js/tile_layers.js correspond to those in TileLayers.py")

    if DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.ESRI_WORLD_IMAGERY:
        bottle.response.status = 400
        return json.dumps({"error": "Cannot currently download with ESRI World Imagery"})
    elif DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.USA_NAIP_LIST:
        naip_data, raster_profile, raster_transform = DataLoader.download_usa_data_by_extent(extent, geo_data_type=DataLoader.GeoDataTypes.NAIP)
    elif DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.CUSTOM:
        dl = DATA_LAYERS[dataset]
        layer = Utils.get_shape_layer_by_name(dl["shapes"], zone_layer_name)

        if layer is None:
            bottle.response.status = 400
            return json.dumps({"error": "You have not selected a set of zones to use"})
        print("Downloading using shapes from layer: %s" % (layer["name"]))
        naip_data, raster_profile, raster_transform = DataLoader.download_custom_data_by_extent(extent, shapes=layer["shapes_geoms"], shapes_crs=layer["shapes_crs"], data_fn=dl["data_fn"])
    naip_data = np.rollaxis(naip_data, 0, 3)


    output = AugmentationState.model.run(naip_data, extent, True)
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
    img_hard = np.round(Utils.class_prediction_to_img(output, True, color_list)*255,0).astype(np.uint8)
    img_hard = cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGR)
    img_hard[nodata_mask] = [0,0,0]
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
    f.write("Class id\tClass name\tFrequency\n")
    for i in range(len(vals)):
        f.write("%d\t%s\t%0.4f%%\n" % (vals[i], name_list[vals[i]], (counts[i] / np.sum(counts))*100))
    f.close()
    data["downloadStatistics"] = "downloads/%s.txt" % (tmp_id)

    bottle.response.status = 200
    return json.dumps(data)


def get_input():
    ''' Method called for POST `/getInput`
    '''
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data) # record this interaction

    # Inputs
    extent = data["extent"]
    dataset = data["dataset"]

    # ------------------------------------------------------
    # Step 1
    #   Transform the input extent into a shapely geometry
    #   Find the tile assosciated with the geometry
    # ------------------------------------------------------
    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------
    if dataset not in DATA_LAYERS:
        raise ValueError("Dataset doesn't seem to be valid, do the datasets in js/tile_layers.js correspond to those in TileLayers.py")

    if DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.ESRI_WORLD_IMAGERY:
        padding = 0.0005
        naip_data, naip_crs, naip_transform, naip_bounds, naip_index = DataLoader.get_esri_data_by_extent(extent, padding=padding)
    elif DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.USA_NAIP_LIST:
        padding = 20
        naip_data, naip_crs, naip_transform, naip_bounds, naip_index = DataLoader.get_usa_data_by_extent(extent, padding=padding, geo_data_type=DataLoader.GeoDataTypes.NAIP)
    elif DATA_LAYERS[dataset]["data_layer_type"] == DataLayerTypes.CUSTOM:
        if "padding" in DATA_LAYERS[dataset]:
            padding = DATA_LAYERS[dataset]["padding"]
        else:
            padding = 20
        naip_data, naip_crs, naip_transform, naip_bounds, naip_index = DataLoader.get_custom_data_by_extent(extent, padding=padding, data_fn=DATA_LAYERS[dataset]["data_fn"])
    naip_data = np.rollaxis(naip_data, 0, 3)
    

    naip_data, new_bounds = DataLoader.warp_data_to_3857(naip_data, naip_crs, naip_transform, naip_bounds)
    naip_data = DataLoader.crop_data_by_extent(naip_data, new_bounds, extent)

    naip_img = naip_data[:,:,:3].copy().astype(np.uint8) # keep the RGB channels to save as a color image

    naip_img = cv2.imencode(".png", cv2.cvtColor(naip_img, cv2.COLOR_RGB2BGR))[1].tostring()
    naip_img = base64.b64encode(naip_img).decode("utf-8")
    data["input_naip"] = naip_img

    bottle.response.status = 200
    return json.dumps(data)

def do_get():
    '''Dummy method for easily testing whether the server is running correctly'''
    return "Backend server running"


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backend Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4444)
    parser.add_argument("--model", action="store", dest="model",
        choices=[
            "cached",
            "iclr_keras",
            "iclr_cntk",
            "nips_sr",
            "existing",
            "nips_hr",
            "group_norm",
        ],
        help="Model to use", required=True
    )
    parser.add_argument("--fine_tune", action="store", dest="fine_tune",
        choices=[
            "last_layer",
            "last_k_layers",
            "group_params",
            "last_k_plus_group_params",
            "group_params_then_last_k"
        ],
        help="Model to use", required=True
    )
    parser.add_argument("--model_fn", action="store", dest="model_fn", type=str, help="Model fn to use", default=None)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", default=0)

    args = parser.parse_args(sys.argv[1:])

    model = None
    if args.model == "cached":
        if args.model_fn not in ["7_10_2018","1_3_2019"]:
            print("When using `cached` model you must specify either '7_10_2018', or '1_3_2019'. Exiting...")
            return
        model = ServerModelsCachedFormat.CachedModel(args.model_fn)
    elif args.model == "iclr_keras":
        model = ServerModelsICLRDynamicFormat.KerasModel(args.model_fn, args.gpuid)
    elif args.model == "iclr_cntk":
        model = ServerModelsICLRFormat.CNTKModel(args.model_fn, args.gpuid)
    elif args.model == "nips_sr":
        if args.fine_tune == "last_layer":
            model = ServerModelsNIPS.KerasDenseFineTune(args.model_fn, args.gpuid, superres=True)
        elif args.fine_tune == "last_k_layers":
            model = ServerModelsNIPS.KerasBackPropFineTune(args.model_fn, args.gpuid, superres=True)
    elif args.model == "nips_hr":
        if args.fine_tune == "last_layer":
            model = ServerModelsNIPS.KerasDenseFineTune(args.model_fn, args.gpuid, superres=False)
        elif args.fine_tune == "last_k_layers":
            model = ServerModelsNIPS.KerasBackPropFineTune(args.model_fn, args.gpuid, superres=False)
    elif args.model == "group_norm":
        if args.fine_tune == "last_k_layers":
            model = ServerModelsNIPSGroupNorm.LastKLayersFineTune(args.model_fn, args.gpuid, last_k_layers=1)
        elif args.fine_tune == "group_params":
            model = ServerModelsNIPSGroupNorm.UnetgnFineTune(args.model_fn, args.gpuid)
        elif args.fine_tune == "last_k_plus_group_params":
            model = ServerModelsNIPSGroupNorm.GroupParamsLastKLayersFineTune(args.model_fn, args.gpuid, last_k_layers=2)
        elif args.fine_tune == "group_params_then_last_k":
            model = ServerModelsNIPSGroupNorm.GroupParamsThenLastKLayersFineTune(args.model_fn, args.gpuid, last_k_layers=2)
    elif args.model == "existing":
        model = joblib.load(args.model_fn)
    else:
        print("Model isn't implemented, aborting")
        return

    AugmentationState.model = model
    AugmentationState.debug_mode = args.debug
    if args.model == "existing":
        AugmentationState.reset(soft=True)
    else:
        AugmentationState.reset()

    # Setup the bottle server 
    app = bottle.Bottle()

    app.add_hook("after_request", enable_cors)
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

    app.route("/heatmap/<z>/<y>/<x>", method="GET", callback=do_heatmap)

    app.route("/", method="GET", callback=do_get)

    bottle_server_kwargs = {
        "host": args.host,
        "port": args.port,
        "debug": args.verbose,
        "server": "tornado",
        "reloader": False # Every time we change something the server will automatically reload. This breaks CNTK.
    }
    app.run(**bottle_server_kwargs)

if __name__ == "__main__":
    main()
