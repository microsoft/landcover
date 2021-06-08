import sys
import os
import time
import datetime
import collections
import subprocess
import shutil

import base64
import json
import uuid
import pickle

import numpy as np

import fiona.transform
import shapely.geometry

import joblib

from .DataLoader import warp_data_to_3857, crop_data_by_extent, crop_data_by_geometry, extent_to_transformed_geom
from .Utils import get_random_string, AtomicCounter

import logging
LOGGER = logging.getLogger("server")

SESSION_BASE_PATH = './tmp/session'
SESSION_FOLDER = SESSION_BASE_PATH + "/" + datetime.datetime.now().strftime('%Y-%m-%d')


def manage_session_folders():
    if not os.path.exists(SESSION_BASE_PATH):
        os.makedirs(SESSION_BASE_PATH)
    if not os.path.exists(SESSION_FOLDER):
        shutil.rmtree(SESSION_BASE_PATH)
        os.makedirs(SESSION_FOLDER)


class Session():

    def __init__(self, session_id, model):
        LOGGER.info("Instantiating a new session object with id: %s" % (session_id))

        self.storage_type = "file" # this will be "table" or "file"
        self.storage_path = "tmp/output/" # this will be a file path
        self.table_service = None # this will be an instance of TableService

        self.model = model
        self.current_transform = ()

        self.current_snapshot_string = get_random_string(8)
        self.current_snapshot_idx = 0
        self.current_request_counter = AtomicCounter()
        self.request_list = []

        self.session_id = session_id
        self.creation_time = time.time()
        self.last_interaction_time = self.creation_time

        self.tile_map = None

        self.mapfile = '/home/kolya/data/saved/' + model.model_fn[-15:] + '.npy'

        self.fill_in_other_year = False

        try:
            self.tile_map = np.load(self.mapfile)
        except: 
            self.tile_map = None

        if self.fill_in_other_year:
            try:
                self.other_tile_map = np.load(self.mapfile[:-5] + ('7' if self.mapfile[-5]=='3' else '3') + '.npy')
                if not (self.tile_map is None):
                    self.tile_map[self.tile_map==255] = self.other_tile_map[self.tile_map==255]
                    del self.other_tile_map
                else:
                    self.tile_map = self.other_tile_map
            except: 
                self.other_tile_map = None
        else: self.other_tile_map = None

    def reset(self, soft=False, from_cached=None):
        if not soft:
            self.model.reset() # can't fail, so don't worry about it
        self.current_snapshot_string = get_random_string(8)
        self.current_snapshot_idx = 0
        self.current_request_counter = AtomicCounter()
        self.request_list = []

        if self.storage_type == "table":
            self.table_service.insert_entity("webtoolsessions",
            {
                "PartitionKey": str(np.random.randint(0,8)),
                "RowKey": str(uuid.uuid4()),
                "session_id": self.current_snapshot_string,
                "server_hostname": os.uname()[1],
                "server_sys_argv": ' '.join(sys.argv),
                "base_model": from_cached
            })

    def load(self, encoded_model_fn):
        model_fn = base64.b64decode(encoded_model_fn).decode('utf-8')

        del self.model
        self.model = joblib.load(model_fn)

    def save(self, model_name):

        if self.storage_type is not None:
            assert self.storage_path is not None # we check for this when starting the program

            snapshot_id = "%s_%d" % (model_name, self.current_snapshot_idx)
            
            LOGGER.info("Saving state for %s" % (snapshot_id))
            base_dir = os.path.join(self.storage_path, self.current_snapshot_string)
            if not os.path.exists(base_dir):
                os.makedirs(base_dir, exist_ok=False)
            
            model_fn = os.path.join(base_dir, "%s_model.p" % (snapshot_id))
            #joblib.dump(self.model, model_fn, protocol=pickle.HIGHEST_PROTOCOL)

            if self.storage_type == "file":
                request_list_fn = os.path.join(base_dir, "%s_request_list.p" % (snapshot_id))
                joblib.dump(self.request_list, request_list_fn, protocol=pickle.HIGHEST_PROTOCOL)
            elif self.storage_type == "table":
                # We don't serialize the request list when saving to table storage
                pass

            self.current_snapshot_idx += 1
            return base64.b64encode(model_fn.encode('utf-8')).decode('utf-8') # this is super dumb
        else:
            return None
    
    def add_entry(self, data):
        data = data.copy()
        data["time"] = datetime.datetime.now()
        data["current_snapshot_index"] = self.current_snapshot_idx
        current_request_counter = self.current_request_counter.increment()
        data["current_request_index"] = current_request_counter

        assert "experiment" in data

        if self.storage_type == "file":
            self.request_list.append(data)
        
        elif self.storage_type == "table":

            data["PartitionKey"] = self.current_snapshot_string
            data["RowKey"] = "%s_%d" % (data["experiment"], current_request_counter)

            for k in data.keys():
                if isinstance(data[k], dict) or isinstance(data[k], list):
                    data[k] = json.dumps(data[k])
            
            try:
                self.table_service.insert_entity("webtoolinteractions", data)
            except Exception as e:
                LOGGER.error(e)
        else:
            # The storage_type / --storage_path command line args were not set
            pass

    def pred_patch(self, patch, dataset_profile, target_extent, patch_transform, idx):
        # need to paste the returned predictions to self.output_map
        outputs = self.model.run(patch, False, idx)
        output = outputs[self.model.current_model_idx]

        # Crop output to extent
        dataset_crs = dataset_profile["crs"].to_string()
        dataset_transform = dataset_profile["transform"]
        dataset_height = dataset_profile["height"]
        dataset_width = dataset_profile["width"]

        cropped_output = crop_data_by_extent(output, dataset_crs, patch_transform, target_extent)[0]

        data_mask = cropped_output.sum(axis=2) > 0        

        transformed_extent = extent_to_transformed_geom(target_extent, dataset_crs)
        transformed_bounds = shapely.geometry.shape(transformed_extent).bounds

        # Save in the correct place
        minx, maxy = ~dataset_transform * (transformed_bounds[0], transformed_bounds[1])
        maxx, miny = ~dataset_transform * (transformed_bounds[2], transformed_bounds[3])
        minx = int(np.floor(minx))
        miny = int(np.floor(miny))
        maxx = int(np.ceil(maxx))
        maxy = int(np.ceil(maxy))
        patch_height = maxy - miny
        patch_width = maxx - minx

        # minx,miny,maxx,maxy can be outside of the tile boundary, we need to figure out the offsets to use
        minx = max(0, minx)
        miny = max(0, miny)
        maxx = min(dataset_width, maxx)
        maxy = min(dataset_height, maxy)

        # Make sure our tile map is setup
        if self.tile_map is None:
            if self.fill_in_other_year and not (self.other_tile_map is None):
                self.tile_map = self.other_tile_map
            else: 
                self.tile_map = np.zeros((dataset_height, dataset_width), dtype=np.uint8)
                self.tile_map[:] = 255
        current_image = self.tile_map[miny:maxy, minx:maxx]
        current_image[data_mask] = cropped_output[data_mask].argmax(1)
        current_image[current_image==20] = 255
        self.tile_map[miny:maxy, minx:maxx] = current_image

        return outputs

    def pred_tile(self, tile, dataset_profile, target_polygon, tile_transform):
        outputs = self.model.run(tile, True, -1)
        output = outputs[self.model.current_model_idx]

        # Crop output to extent
        dataset_crs = dataset_profile["crs"].to_string()
        dataset_transform = dataset_profile["transform"]
        dataset_height = dataset_profile["height"]
        dataset_width = dataset_profile["width"]

        transformed_target_shape = fiona.transform.transform_geom("epsg:4326", dataset_crs, target_polygon)
        cropped_output = crop_data_by_geometry(output, dataset_crs, tile_transform, transformed_target_shape, dataset_crs)[0]

        transformed_bounds = shapely.geometry.shape(transformed_target_shape).bounds

        data_mask = cropped_output.sum(axis=2) > 0

        # Save in the correct place
        minx, maxy = ~dataset_transform * (transformed_bounds[0], transformed_bounds[1])
        maxx, miny = ~dataset_transform * (transformed_bounds[2], transformed_bounds[3])
        minx = int(np.floor(minx))
        miny = int(np.floor(miny))
        maxx = int(np.ceil(maxx))
        maxy = int(np.ceil(maxy))
        patch_height = maxy - miny
        patch_width = maxx - minx

        # minx,miny,maxx,maxy can be outside of the tile boundary, we need to figure out the offsets to use
        minx = max(0, minx)
        miny = max(0, miny)
        maxx = min(dataset_width, maxx)
        maxy = min(dataset_height, maxy)

        # Make sure our tile map is setup
        if self.tile_map is None:
            if self.fill_in_other_year and not (self.other_tile_map is None):
                self.tile_map = self.other_tile_map
            else: 
                self.tile_map = np.zeros((dataset_height, dataset_width), dtype=np.uint8)
                self.tile_map[:] = 255
        current_image = self.tile_map[miny:maxy, minx:maxx]
        current_image[data_mask] = cropped_output[data_mask].argmax(1)
        current_image[current_image==20] = 255
        self.tile_map[miny:maxy, minx:maxx] = current_image

        return output

    def get_tile_predictions(self):
        print('saving')
        if not (self.tile_map is None): np.save(self.mapfile, self.tile_map)
        print('saved')
        return self.tile_map
