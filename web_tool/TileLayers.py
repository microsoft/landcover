import os

import fiona
import shapely.geometry
from enum import Enum

from web_tool.frontend_server import ROOT_DIR

def load_geojson_as_list(fn):
    shapes = []
    crs = None
    with fiona.open(fn) as f:
        crs = f.crs
        for row in f:
            shape = shapely.geometry.shape(row["geometry"])
            shapes.append(shape)
    return shapes, crs

class DataLayerTypes(Enum):
    ESRI_WORLD_IMAGERY = 1
    USA_NAIP_LIST = 2
    CUSTOM = 3

'''
This dictionary defines how the backend tool will return data to the frontend.

An entry is formated like below:

"LAYER NAME": {
    "data_layer_type": DataLayerTypes.ESRI_WORLD_IMAGERY,
    "shapes_fn": None,
    "data_fn": None,
    "shapes": None,  # NOTE: this is always `None` and populated automatically when this file loads (see code at bottom of file)
    "shapes_crs": None  # NOTE: this is always `None` and populated automatically when this file loads (see code at bottom of file)
    "padding": None # NOTE: this is optional and only used in DataLayerTypes.CUSTOM
}

LAYER_NAME - should correspond to an entry in js/tile_layers.js
data_layer_type -  should be an item from the DataLayerTypes enum and describes where the data comes from.
  - If ESRI_WORLD_IMAGERY then the backend will lookup imagery from the ESRI World Imagery basemap and not respond to requests for downloading
  - If USA_NAIP_LIST then the backend will lookup imagery from the full USA tile_index (i.e. how we usually do it) and requests for downloading will be executed on the same tiles
  - If CUSTOM then the backend will query the "shapes_fn" and "data_fn" files for how/what to download, downloading will happen similarlly
shapes_fn - should be a path, relative to `frontend_server.ROOT_DIR`, of a geojson defining shapes over which the data_fn file is valid. When a "download" happens the raster specified by "data_fn" will be masked with one of these shapes.
data_fn - should be a path, relative to `frontend_server.ROOT_DIR`, of a raster file defining the imagery to use
shapes - list of `shapely.geometry.shape` objects created from the shapes in "shapes_fn".
shapes_crs - the CRS of the shapes_fn
padding - NOTE: Optional, only used in DataLayerTypes.CUSTOM - defines the padding used in raster extraction, used to get the required 240x240 input
'''
DATA_LAYERS = {
    "esri_world_imagery": { 
        "data_layer_type": DataLayerTypes.ESRI_WORLD_IMAGERY,
        "shapes": None,
        "data_fn": None,
    },
    "esri_world_imagery_naip": { 
        "data_layer_type": DataLayerTypes.USA_NAIP_LIST,
        "shapes": None,
        "data_fn": None,
    },
    "osm": {
        "data_layer_type": DataLayerTypes.ESRI_WORLD_IMAGERY,
        "shapes": None,
        "data_fn": None,
    },
    "chesapeake": {
        "data_layer_type": DataLayerTypes.USA_NAIP_LIST,
        "shapes": None,
        "data_fn": None,
    },
    "demo_set_1": {
        "data_layer_type": DataLayerTypes.USA_NAIP_LIST,
        "shapes": None,
        "data_fn": None,
    },
    "user_study_1": {
        "data_layer_type": DataLayerTypes.USA_NAIP_LIST,
        "shapes": None,
        "data_fn": None,
    },
    "user_study_2": {
        "data_layer_type": DataLayerTypes.USA_NAIP_LIST,
        "shapes": None,
        "data_fn": None,
    },
    "user_study_3": {
        "data_layer_type": DataLayerTypes.USA_NAIP_LIST,
        "shapes": None,
        "data_fn": None,
    },
    "user_study_4": {
        "data_layer_type": DataLayerTypes.USA_NAIP_LIST,
        "shapes": None,
        "data_fn": None,
    },
    "user_study_5": {
        "data_layer_type": DataLayerTypes.CUSTOM,
        "shapes": [
            {"name": "Area boundary", "shapes_fn": "shapes/user_study_5_outline.geojson", "zone_name_key": None}
        ],
        "data_fn": "tiles/user_study_5.tif",
        "padding": 20
    },
    "philipsburg_mt": {
        "data_layer_type": DataLayerTypes.USA_NAIP_LIST,
        "shapes": None,
        "data_fn": None,
    },
    "aceh": {
        "data_layer_type": DataLayerTypes.CUSTOM,
        "shapes": None,
        "data_fn": None,
    },
    "yangon_sentinel": {
        "data_layer_type": DataLayerTypes.CUSTOM,
        "shapes": [
            {"name": "Admin 1", "shapes_fn": "shapes/yangon_sentinel_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Admin 2", "shapes_fn": "shapes/yangon_sentinel_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Admin 3", "shapes_fn": "shapes/yangon_sentinel_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "data_fn": "tiles/yangon.tif",
        "padding": 1100
    },
    "hcmc_sentinel": {
        "data_layer_type": DataLayerTypes.CUSTOM,
        "shapes": [
            {"name": "Admin 1", "shapes_fn": "shapes/hcmc_sentinel_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Admin 2", "shapes_fn": "shapes/hcmc_sentinel_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Admin 3", "shapes_fn": "shapes/hcmc_sentinel_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "data_fn": "tiles/hcmc_sentinel.tif",
        "padding": 1100
    },
    "yangon_lidar": {
        "data_layer_type": DataLayerTypes.CUSTOM,
        "shapes": [
            {"name": "Admin 1", "shapes_fn": "shapes/yangon_lidar_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Admin 2", "shapes_fn": "shapes/yangon_lidar_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Admin 3", "shapes_fn": "shapes/yangon_lidar_admin_3_clipped.geojson", "zone_name_key": "NAME_3"},
            {"name": "Admin 4", "shapes_fn": "shapes/yangon_lidar_admin_4_clipped.geojson", "zone_name_key": "Ward"}
        ],
        "data_fn": "tiles/yangon_lidar.tif",
    },
    "hcmc_dg": {
        "data_layer_type": DataLayerTypes.CUSTOM,
        "shapes": [
            {"name": "Admin 1", "shapes_fn": "shapes/hcmc_digital-globe_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Admin 2", "shapes_fn": "shapes/hcmc_digital-globe_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Admin 3", "shapes_fn": "shapes/hcmc_digital-globe_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "data_fn": "tiles/HCMC.tif",
        "padding": 0
    },
    "airbus": {
        "data_layer_type": DataLayerTypes.CUSTOM,
        "shapes": [
            {"name": "Grid", "shapes_fn": "shapes/airbus-data-grid-epsg4326.geojson", "zone_name_key": None}
        ],
        "data_fn": "tiles/airbus_epsg4326.tif",
        "padding": 0.003
    }
}

for k in DATA_LAYERS.keys():
    if DATA_LAYERS[k]["shapes"] is not None:
        print("Loading shapes for the %s dataset" % (k))
        for zone_layer in DATA_LAYERS[k]["shapes"]:
            fn = os.path.join(ROOT_DIR, zone_layer["shapes_fn"])
            if os.path.exists(fn):
                shapes, crs = load_geojson_as_list(fn)
                zone_layer["shapes_geoms"] = shapes
                zone_layer["shapes_crs"] = crs["init"]
            else:
                print("WARNING: %s doesn't exist, this server will not be able to serve the '%s' dataset" % (fn, k))
