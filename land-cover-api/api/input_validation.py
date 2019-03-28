from flask import Flask, request, abort
from enum import Enum

import json

class InputType(Enum):
    latlon = 1
    extent = 2

class RequestType(Enum):
    classify = 1
    tile = 2
   
VALUE_NOT_PROVIDED = " value not provided"
SHOULD_BE_NUMERIC = " should be numeric"
INVALID_JSON_ERROR = "Unable to parse the request body. Please request with valid json"

class InputValidator:
    
    def validate_input_data(self, request_data, input_type, request_type):  
        try:
            data = json.loads(request_data)
        except Exception as e:
            return False, INVALID_JSON_ERROR
        
        if(len(data) == 0):
            return False, INVALID_JSON_ERROR
            
        #convert all json to lowercase, makes it easier to check
        data = eval(repr(data).lower())

        if(request_type == RequestType.classify):
            if not "weights" in data:
                return False, "weights" + VALUE_NOT_PROVIDED

        if(input_type == InputType.latlon):
            input_keys = ["lat", "lon", "patchsize", "latestwkid"]

            for key in input_keys:
                if not key in data:
                    return False, key + VALUE_NOT_PROVIDED
                else:
                    parameter = data[key]
                    if not (isinstance(parameter, float) or isinstance(parameter, int)):
                        return False, key + SHOULD_BE_NUMERIC
        elif(input_type == InputType.extent):
            input_keys = ["xmin", "xmax", "ymin", "ymax", "latestwkid"]
            
            data = data['extent']
            
            for key in input_keys:
                if key == "latestwkid":
                    if not key in data["spatialreference"]:
                        return False, key + VALUE_NOT_PROVIDED
                    parameter = data["spatialreference"][key]

                else:
                    if not key in data:
                        return False, key + VALUE_NOT_PROVIDED
                    parameter = data[key]
                
                if not (isinstance(parameter, float) or isinstance(parameter, int)):
                    return False, key + SHOULD_BE_NUMERIC

        return True, "All inputs are valid"          
