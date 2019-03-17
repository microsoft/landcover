#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
import sys
import os
import time

import bottle
import argparse
import functools
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

import ServerModelsICLRFormat, ServerModelsCachedFormat, ServerModelsICLRDynamicFormat

from sklearn.neural_network import MLPClassifier

current_transform = None
current_predictions = None
current_features = None

correction_pts = []
corrections_x = []
corrections_y = []
augment_model = MLPClassifier(
    hidden_layer_sizes=(),
    activation='relu',
    alpha=0.001,
    solver='lbfgs',
    verbose=True,
    validation_fraction=0.0,
    n_iter_no_change=10
)
model_fit = False

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def get_random_string(length):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return ''.join([alphabet[np.random.randint(0, len(alphabet))] for i in range(length)])

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

def save_model():
    global corrections_x, corrections_y, correction_pts, augment_model, model_fit
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    if model_fit:
        snapshot_id = get_random_string(8)
        data["message"] = "Saved model '%s'" % (snapshot_id)
        data["success"] = True

        print("Saving snapshot %s" % (snapshot_id))
        np.save("output/%s_x.npy" % (snapshot_id), corrections_x)
        np.save("output/%s_y.npy" % (snapshot_id), corrections_y)
        joblib.dump(correction_pts, "output/%s_pts.p" % (snapshot_id), protocol=pickle.HIGHEST_PROTOCOL)
        joblib.dump(augment_model, "output/%s_x.model" % (snapshot_id), protocol=pickle.HIGHEST_PROTOCOL)

    else:
        data["message"] = "There is not a trained model to save"
        data["success"] = False


    bottle.response.status = 200
    return json.dumps(data)

def reset_model():
    global corrections_x, corrections_y, augment_model, model_fit
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    corrections_x = []
    corrections_y = []
    augment_model = MLPClassifier(
        hidden_layer_sizes=(),
        activation='relu',
        alpha=0.001,
        solver='lbfgs',
        verbose=True,
        validation_fraction=0.0,
        n_iter_no_change=10
    )
    model_fit = False

    data["message"] = "Reset model"
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)

def retrain_model():
    global corrections_x, corrections_y, augment_model, model_fit
    bottle.response.content_type = 'application/json'
    data = bottle.request.json

    if len(corrections_x) > 0:
        x_train = np.concatenate(corrections_x, axis=0)
        y_train = np.concatenate(corrections_y, axis=0)

        print(x_train.shape, y_train.shape)

        vals, counts = np.unique(y_train, return_counts=True)
        print(list(zip(vals, counts)))

        if len(vals) == 4:
            print("Fitting model with %d samples" % (x_train.shape[0]))
            augment_model.fit(x_train, y_train)
            model_fit = True
            print("Finished fitting model")

            data["message"] = "Fit accessory model with %d samples" % (x_train.shape[0])
            data["success"] = True
        else:
            data["message"] = "Need to include training samples from each class"
            data["success"] = False
    else:
        data["message"] = "No training data submitted yet"
        data["success"] = False

    # Perform validation set testing
    # if model_fit and True:
    #     # Test on all data
    #     y_pred = augment_model.predict(ServerModelsKDD.x_test)
    #     print(y_pred.shape)
    #     print(ServerModelsKDD.y_test.shape)
    #     print("acc", np.sum(y_pred == ServerModelsKDD.y_test) / y_pred.shape[0])
    #     print("corrected", 10371631 - np.sum(ServerModelsKDD.y_test != y_pred))
    #     print("\% for graph",  (10371631 - np.sum(ServerModelsKDD.y_test != y_pred)) / 10371631)


    bottle.response.status = 200
    return json.dumps(data)

def record_correction():
    global current_transform, current_predictions, current_features, corrections_x, corrections_y
    bottle.response.content_type = 'application/json'
    data = bottle.request.json


    data["time"] = time.ctime()
    correction_pts.append(dict(data))

    tlat, tlon = data["extent"]["ymax"], data["extent"]["xmin"]
    blat, blon = data["extent"]["ymin"], data["extent"]["xmax"]
    value = data["value"]

    src_crs, dst_crs, dst_transform, rev_dst_transform, padding = current_transform
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

    x_train = current_features[tdst_row:bdst_row+1, tdst_col:bdst_col+1, :].copy().reshape(-1, current_features.shape[2])
    y_train = np.zeros((x_train.shape[0]), dtype=np.uint8)
    y_train[:] = class_idx

    current_predictions[tdst_row:bdst_row+1, tdst_col:bdst_col+1, :] = 0
    current_predictions[tdst_row:bdst_row+1, tdst_col:bdst_col+1, class_idx] = 1

    img_soft = np.round(utils.class_prediction_to_img(current_predictions, False)*255,0).astype(np.uint8)
    img_soft = cv2.imencode(".png", cv2.cvtColor(img_soft, cv2.COLOR_RGB2BGR))[1].tostring()
    img_soft = base64.b64encode(img_soft).decode("utf-8")
    data["output_soft"] = img_soft

    img_hard = np.round(utils.class_prediction_to_img(current_predictions, True)*255,0).astype(np.uint8)
    img_hard = cv2.imencode(".png", cv2.cvtColor(img_hard, cv2.COLOR_RGB2BGR))[1].tostring()
    img_hard = base64.b64encode(img_hard).decode("utf-8")
    data["output_hard"] = img_hard

    corrections_x.append(x_train)
    corrections_y.append(y_train)

    data["message"] = "Successfully submitted correction"
    data["success"] = True

    bottle.response.status = 200
    return json.dumps(data)

def pred_patch(model):
    global augment_model, model_fit, current_transform, current_predictions, current_features
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

    naip_data, padding, transform = DataLoader.get_data_by_extent(naip_fn, extent, DataLoader.GeoDataTypes.NAIP, return_transforms=True)
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
    (output, output_features), name = model.run(naip_data, naip_fn, extent, padding)
    assert output.shape[2] == 4, "The model function should return an image shaped as (height, width, num_classes)"
    output *= weights[np.newaxis, np.newaxis, :] # multiply by the weight vector
    sum_vals = output.sum(axis=2) # need to normalize sums to 1 in order for the rendered output to be correct
    output = output / (sum_vals[:,:,np.newaxis] + 0.000001)
    
    if padding > 0:
        output = output[padding:-padding,padding:-padding,:]
        output_features = output_features[padding:-padding,padding:-padding,:]

    if model_fit:
        print("Augmenting output")
        original_shape = output.shape
        output = output_features.reshape(-1, output_features.shape[2])
        output = augment_model.predict_proba(output)
        output = output.reshape(original_shape)
    
    current_features = output_features.copy()
    current_predictions = output.copy()
    current_transform = transform

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


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backend Server")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debugging", default=False)
    parser.add_argument("--host", action="store", dest="host", type=str, help="Host to bind to", default="0.0.0.0")
    parser.add_argument("--port", action="store", dest="port", type=int, help="Port to listen on", default=4444)
    parser.add_argument("--model", action="store", dest="model", choices=["cached", "keras", "iclr"], help="Model to use", required=True)
    parser.add_argument("--model_fn", action="store", dest="model_fn", type=str, help="Model fn to use", default=None)
    parser.add_argument("--gpu", action="store", dest="gpuid", type=int, help="GPU to use", default=0)

    args = parser.parse_args(sys.argv[1:])

    model = None
    if args.model == "cached":
        if args.model_fn not in ["7_10_2018","1_3_2019"]:
            print("When using `cached` model you must specify either '7_10_2018', or '1_3_2019'. Exiting...")
            return
        model = ServerModelsCachedFormat.CachedModel(args.model_fn)
    elif args.model == "keras":
        model = ServerModelsICLRDynamicFormat.KerasModel(args.model_fn, args.gpuid)
    elif args.model == "iclr":
        model = ServerModelsICLRFormat.CNTKModel(args.model_fn, args.gpuid)
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

    app.route("/recordCorrection", method="OPTIONS", callback=do_options)
    app.route('/recordCorrection', method="POST", callback=record_correction)

    app.route("/retrainModel", method="OPTIONS", callback=do_options)
    app.route('/retrainModel', method="POST", callback=retrain_model)

    app.route("/resetModel", method="OPTIONS", callback=do_options)
    app.route('/resetModel', method="POST", callback=reset_model)

    app.route("/saveModel", method="OPTIONS", callback=do_options)
    app.route('/saveModel', method="POST", callback=save_model)


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
