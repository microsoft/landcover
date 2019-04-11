#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import sys
import time

import bottle
import argparse
import base64
import json

import numpy as np
import cv2

import fiona
import fiona.transform

import DataLoader
import GeoTools
import utils

import pickle
import joblib

import ServerModelsICLRFormat, ServerModelsCachedFormat, ServerModelsICLRDynamicFormat, ServerModelsNIPS, ServerModelsNIPSGroupNorm


def get_random_string(length):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return ''.join([alphabet[np.random.randint(0, len(alphabet))] for i in range(length)])

class AugmentationState():
    current_snapshot_string = "%s_" + get_random_string(8) + "_%d"
    current_snapshot_idx = 0
    model = None

    current_transform = None
    current_naip = None
    current_output = None

    request_list = []

    @staticmethod
    def reset():
        AugmentationState.model.reset() # can't fail, so don't worry about it
        AugmentationState.current_snapshot_string = "%s_" + get_random_string(8) + "_%d"
        AugmentationState.current_snapshot_idx = 0
        AugmentationState.request_list = []

    @staticmethod
    def save(model_name):
        snapshot_id = AugmentationState.current_snapshot_string % (model_name, AugmentationState.current_snapshot_idx)

        print("Saving state for %s" % (snapshot_id))
        joblib.dump(AugmentationState.model, "web-tool/output/%s_model.p" % (snapshot_id), protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(AugmentationState.request_list, "web-tool/output/%s_request_list.p" % (snapshot_id), protocol=pickle.HIGHEST_PROTOCOL)
        
        AugmentationState.current_snapshot_idx += 1

        # TODO: Save other stuff
        '''
        print("Saving snapshot %s" % (snapshot_id))

        os.makedirs("output/", exist_ok=True)
        np.save("output/%s_x.npy" % (snapshot_id), correction_features)
        np.save("output/%s_y.npy" % (snapshot_id), correction_targets)

        np.save("output/%s_base_y.npy" % (snapshot_id), correction_model_predictions)
        np.save("output/%s_sizes.npy" % (snapshot_id), correction_sizes)

        joblib.dump(correction_json, "output/%s_pts.p" % (snapshot_id), protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(augment_model, "output/%s.model" % (snapshot_id), protocol=pickle.HIGHEST_PROTOCOL)

        '''
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

def reset_model():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data)

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
    AugmentationState.request_list.append(data)

    success, message = AugmentationState.model.retrain()

    if success:
        bottle.response.status = 200
        AugmentationState.save(data["experiment"])
    else:
        bottle.response.status = 500

    data["message"] = message
    data["success"] = success

    return json.dumps(data)

def record_correction():
    bottle.response.content_type = 'application/json'
    data = bottle.request.json
    data["time"] = time.ctime()
    AugmentationState.request_list.append(data)


    tlat, tlon = data["extent"]["ymax"], data["extent"]["xmin"]
    blat, blon = data["extent"]["ymin"], data["extent"]["xmax"]
    value = data["value"] # what we want to switch the class to

    src_crs, dst_crs, dst_transform, rev_dst_transform, padding = AugmentationState.current_transform
    #src_crs = "epsg:%s" % (src_crs) # Currently src_crs will be a string like 'epsg:####', this might change with different versions of rasterio --Caleb
    origin_crs = "epsg:%d" % (data["extent"]["spatialReference"]["latestWkid"])

    xs, ys = fiona.transform.transform(origin_crs, src_crs, [tlon,blon], [tlat,blat])
    xs, ys = fiona.transform.transform(src_crs, dst_crs, xs, ys)
    
    tdst_x = xs[0]
    tdst_y = ys[0]
    tdst_col, tdst_row = rev_dst_transform * (tdst_x, tdst_y)
    tdst_row = int(np.floor(tdst_row))
    tdst_col = int(np.floor(tdst_col))

    bdst_x = xs[1]
    bdst_y = ys[1]
    bdst_col, bdst_row = rev_dst_transform * (bdst_x, bdst_y)
    bdst_row = int(np.floor(bdst_row))
    bdst_col = int(np.floor(bdst_col))

    value_to_class_idx = {
        "water" : 0,
        "forest" : 1,
        "field" : 2,
        "built" : 3,
    }
    class_idx = value_to_class_idx[value]

    tdst_row, bdst_row = min(tdst_row, bdst_row)-padding, max(tdst_row, bdst_row)-padding
    tdst_col, bdst_col = min(tdst_col, bdst_col)-padding, max(tdst_col, bdst_col)-padding


    y_pred = AugmentationState.current_output[tdst_row:bdst_row+1, tdst_col:bdst_col+1, :].copy().reshape(-1, AugmentationState.current_output.shape[2])
    '''
    x_train = AugmentationState.current_features[tdst_row:bdst_row+1, tdst_col:bdst_col+1, :].copy().reshape(-1, current_features.shape[2])

    y_train = np.zeros((x_train.shape[0]), dtype=np.uint8)
    y_train[:] = class_idx

    current_predictions[tdst_row:bdst_row+1, tdst_col:bdst_col+1, :] = 0
    current_predictions[tdst_row:bdst_row+1, tdst_col:bdst_col+1, class_idx] = 1
    '''

    AugmentationState.model.add_sample(tdst_row, bdst_row, tdst_col, bdst_col, class_idx)

    '''
    img_soft = np.round(utils.class_prediction_to_img(current_predictions, False)*255,0).astype(np.uint8)
    img_soft = cv2.imencode(".png", cv2.cvtColor(img_soft, cv2.COLOR_RGB2BGR))[1].tostring()
    img_soft = base64.b64encode(img_soft).decode("utf-8")
    data["output_soft"] = img_soft

    img_hard = np.round(utils.class_prediction_to_img(current_predictions, True)*255,0).astype(np.uint8)
    img_hard = cv2.imencode(".png", cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGR))[1].tostring()
    img_hard = base64.b64encode(img_hard).decode("utf-8")
    data["output_hard"] = img_hard
    '''

    data["message"] = "Successfully submitted correction"
    data["count"] = y_pred.shape[0]
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)

def pred_patch():
    ''' Method called for POST `/predPatch`'''
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
        naip_file_name = DataLoader.lookup_tile_by_geom(geom)
    except ValueError as e:
        print(e)
        bottle.response.status = 400
        return json.dumps({"error": str(e)})

    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------

    naip_data, padding, transform = DataLoader.get_data_by_extent(naip_file_name, extent, DataLoader.GeoDataTypes.NAIP, return_transforms=True)
    naip_data = np.rollaxis(naip_data, 0, 3) # we do this here instead of get_data_by_extent because not all GeoDataTypes will have a channel dimension
    
    # record what is going on incase a fine-tuning method needs to use it
    AugmentationState.current_naip = naip_data[padding:-padding,padding:-padding,:].copy()
    AugmentationState.current_transform = transform

    # ------------------------------------------------------
    # Step 3
    #   Run a model on the input data
    #   Apply reweighting
    #   Fix padding
    # ------------------------------------------------------
    output = AugmentationState.model.run(naip_data, naip_file_name, extent, padding)
    assert output.shape[2] == 4, "The model function should return an image shaped as (height, width, num_classes)"
    
    if padding > 0:
        output = output[padding:-padding,padding:-padding,:]
    AugmentationState.current_output = output.copy()

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
        naip_file_name = DataLoader.lookup_tile_by_geom(geom)
    except ValueError as e:
        print(e)
        bottle.response.status = 400
        return json.dumps({"error": str(e)})

    # ------------------------------------------------------
    # Step 2
    #   Load the input data sources for the given tile  
    # ------------------------------------------------------

    naip_data, padding = DataLoader.get_data_by_extent(naip_file_name, extent, DataLoader.GeoDataTypes.NAIP)
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


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backend Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4444)
    parser.add_argument("--model", action="store", dest="model",
        choices=[
            "cached",
            "iclr_keras",
            "iclr_cntk",
            "nips_sr",
            "nips_hr",
            "nips_gn"
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
        model = ServerModelsNIPS.KerasBackPropFineTune(args.model_fn, args.gpuid, superres=True)
    elif args.model == "nips_hr":
        model = ServerModelsNIPS.KerasBackPropFineTune(args.model_fn, args.gpuid, superres=False)
    elif args.model == "nips_gn":
        model = ServerModelsNIPSGroupNorm.UnetgnFineTune(args.model_fn, args.gpuid)
    else:
        print("Model isn't implemented, aborting")
        return

    AugmentationState.model = model

    # Setup the bottle server 
    app = bottle.Bottle()

    app.add_hook("after_request", enable_cors)
    app.route("/predPatch", method="OPTIONS", callback=do_options)
    app.route('/predPatch', method="POST", callback=pred_patch)
    
    app.route("/getInput", method="OPTIONS", callback=do_options)
    app.route('/getInput', method="POST", callback=get_input)

    app.route("/recordCorrection", method="OPTIONS", callback=do_options)
    app.route('/recordCorrection', method="POST", callback=record_correction)

    app.route("/retrainModel", method="OPTIONS", callback=do_options)
    app.route('/retrainModel', method="POST", callback=retrain_model)

    app.route("/resetModel", method="OPTIONS", callback=do_options)
    app.route('/resetModel', method="POST", callback=reset_model)

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
