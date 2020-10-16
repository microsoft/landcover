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
from .DataLoader import DataLoaderCustom, DataLoaderUSALayer, DataLoaderLCLayer, DataLoaderBasemap

def _load_dataset(dataset):
    # Step 1: make sure the dataLayer exists
    if dataset["dataLayer"]["type"] == "CUSTOM":
        fn = dataset["dataLayer"]["path"]
        if not os.path.exists(fn):
            LOGGER.warning("Step 2 failed in loading dataset {}".format(dataset["metadata"]["displayName"]))
            return False # TODO: maybe we should make these errors more descriptive (explain why we can't load a dataset)

    # Step 2: setup the appropriate DatasetLoader
    if dataset["dataLayer"]["type"] == "CUSTOM":
        data_loader = DataLoaderCustom(**dataset["dataLayer"])
    elif dataset["dataLayer"]["type"] == "USA_LAYER":
        data_loader = DataLoaderUSALayer(**dataset["dataLayer"])
    elif dataset["dataLayer"]["type"] == "LC_LAYER":
        data_loader = DataLoaderLCLayer(**dataset["dataLayer"])
    elif dataset["dataLayer"]["type"] == "BASEMAP":
        data_loader = DataLoaderBasemap(**dataset["dataLayer"])
    else:
        LOGGER.warning("Step 3 failed in loading dataset {}".format(dataset["metadata"]["displayName"]))
        return False # TODO: maybe we should make these errors more descriptive (explain why we can't load a dataset)

    return data_loader

def load_datasets():
    """Returns a dictionary of key:value where keys are dataset names (from "datasets.json" / "datasets.mine.json") and values are instances of classes that extend DataLoaderAbstract
    """
    datasets = dict()

    dataset_json = json.load(open(os.path.join(ROOT_DIR, "datasets.json"),"r"))
    for key, dataset in dataset_json.items():
        data_loader = _load_dataset(dataset)

        if data_loader is False:
            LOGGER.warning("Files are missing, we will not be able to serve the following dataset: '%s'" % (key)) 
        else:
            datasets[key] = data_loader

    if os.path.exists(os.path.join(ROOT_DIR, "datasets.mine.json")):
        dataset_json = json.load(open(os.path.join(ROOT_DIR, "datasets.mine.json"),"r"))
        for key, dataset in dataset_json.items():

            if key not in datasets:
                data_loader = _load_dataset(dataset)

                if data_loader is False:
                    LOGGER.warning("Files are missing, we will not be able to serve the following dataset: '%s'" % (key)) 
                else:
                    datasets[key] = data_loader
            else:
                LOGGER.warning("There is a conflicting dataset key in datasets.mine.json, skipping.")

    return datasets

def is_valid_dataset(dataset_key):
    dataset_json = json.load(open(os.path.join(ROOT_DIR, "datasets.json"), "r"))
    if os.path.exists(os.path.join(ROOT_DIR, "datasets.mine.json")):
        dataset_mine_json = json.load(open(os.path.join(ROOT_DIR, "datasets.mine.json"), "r"))

    return (dataset_key in dataset_json) or (dataset_key in dataset_mine_json)