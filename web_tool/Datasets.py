import os
import json

import utm
import fiona
import fiona.transform
import shapely
import shapely.geometry

import logging
LOGGER = logging.getLogger("server")

from . import ROOT_DIR
from .DataLoader import DataLoaderCustom, DataLoaderUSALayer, DataLoaderBasemap


def get_area_from_geometry(geom, src_crs="epsg:4326"):
    if geom["type"] == "Polygon":
        lon, lat = geom["coordinates"][0][0]
    elif geom["type"] == "MultiPolygon":
        lon, lat = geom["coordinates"][0][0][0]
    else:
        raise ValueError("Polygons and MultiPolygons only")

    zone_number = utm.latlon_to_zone_number(lat, lon)
    hemisphere = "+north" if lat > 0 else "+south"
    dest_crs = "+proj=utm +zone=%d %s +datum=WGS84 +units=m +no_defs" % (zone_number, hemisphere)
    projected_geom = fiona.transform.transform_geom(src_crs, dest_crs, geom)
    area = shapely.geometry.shape(projected_geom).area / 1000000.0 # we calculate the area in square meters then convert to square kilometers
    return area

def _load_geojson_as_list(fn):
    ''' Takes a geojson file as input and outputs a list of shapely `shape` objects in that file and their corresponding areas in km^2.

    We calculate area here by re-projecting the shape into its local UTM zone, converting it to a shapely `shape`, then using the `.area` property.
    '''
    shapes = []
    areas = []
    crs = None
    with fiona.open(fn) as f:
        src_crs = f.crs
        for row in f:
            geom = row["geometry"]
            
            area = get_area_from_geometry(geom, src_crs)
            areas.append(area)

            shape = shapely.geometry.shape(geom)
            shapes.append(shape)
    return shapes, areas, src_crs


def _load_dataset(dataset):
    # Step 1: load the shape layers
    shape_layers = {}
    if dataset["shapeLayers"] is not None:
        for shape_layer in dataset["shapeLayers"]:
            fn = shape_layer["shapesFn"]
            if os.path.exists(fn):
                shapes, areas, crs = _load_geojson_as_list(fn)
                
                shape_layer["geoms"] = shapes
                shape_layer["areas"] = areas
                shape_layer["crs"] = crs["init"] # TODO: will this break with fiona version; I think `.crs` will turn into a PyProj object
                shape_layers[shape_layer["name"]] = shape_layer
            else:
                return False # TODO: maybe we should make these errors more descriptive (explain why we can't load a dataset)

    # Step 2: make sure the dataLayer exists
    if dataset["dataLayer"]["type"] == "CUSTOM":
        fn = dataset["dataLayer"]["path"]
        if not os.path.exists(fn):
            return False # TODO: maybe we should make these errors more descriptive (explain why we can't load a dataset)

    # Step 3: setup the appropriate DatasetLoader
    if dataset["dataLayer"]["type"] == "CUSTOM":
        data_loader = DataLoaderCustom(dataset["dataLayer"]["path"], shape_layers, dataset["dataLayer"]["padding"])
    elif dataset["dataLayer"]["type"] == "USA_LAYER":
        data_loader = DataLoaderUSALayer(shape_layers, dataset["dataLayer"]["padding"])
    elif dataset["dataLayer"]["type"] == "BASEMAP":
        data_loader = DataLoaderBasemap(dataset["dataLayer"]["path"], dataset["dataLayer"]["padding"])
    else:
        return False # TODO: maybe we should make these errors more descriptive (explain why we can't load a dataset)

    return {
        "data_loader": data_loader,
        "shape_layers": shape_layers,
    }

def load_datasets():
    datasets = dict()

    dataset_json = json.load(open(os.path.join(ROOT_DIR, "datasets.json"),"r"))
    for key, dataset in dataset_json.items():
        dataset_object = _load_dataset(dataset)
        
        if dataset_object is False:
            LOGGER.warning("Files are missing, we will not be able to serve the following dataset: '%s'" % (key)) 
        else:
            datasets[key] = dataset_object

    if os.path.exists(os.path.join(ROOT_DIR, "datasets.mine.json")):
        dataset_json = json.load(open(os.path.join(ROOT_DIR, "datasets.mine.json"),"r"))
        for key, dataset in dataset_json.items():

            if key not in datasets:
                dataset_object = _load_dataset(dataset)
                
                if dataset_object is False:
                    LOGGER.warning("Files are missing, we will not be able to serve the following dataset: '%s'" % (key)) 
                else:
                    datasets[key] = dataset_object
            else:
                LOGGER.warning("There is a conflicting dataset key in datasets.mine.json, skipping.")

    return datasets

def is_valid_dataset(dataset_key):

    dataset_json = json.load(open(os.path.join(ROOT_DIR, "datasets.json"), "r"))
    if os.path.exists(os.path.join(ROOT_DIR, "datasets.mine.json")):
        dataset_mine_json = json.load(open(os.path.join(ROOT_DIR, "datasets.mine.json"), "r"))

    return (dataset_key in dataset_json) or (dataset_key in dataset_mine_json)