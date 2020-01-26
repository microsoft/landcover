import os
import json

from web_tool import ROOT_DIR

_MODEL_FN = "models.json"


def _load_model(model):
    if not os.path.exists(model["model"]["fn"]):
        return False
    return {
        "fn": model["model"]["fn"],
        "fine_tune_layer": model["model"]["fineTuneLayer"]
    }

def load_models():
    model_json = json.load(open(os.path.join(ROOT_DIR,_MODEL_FN),"r"))
    models = dict()

    for key, model in model_json.items():
        model_object = _load_model(model)
        
        if model_object is False:
            print("WARNING: files are missing, we will not be able to serve '%s' model" % (key)) 
        else:
            models[key] = model_object

    return models