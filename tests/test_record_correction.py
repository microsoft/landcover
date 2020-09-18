import sys
sys.path.append("..")

import numpy as np

import rasterio
import fiona
import fiona.transform

from web_tool.DataLoader import DataLoaderCustom

pred_patch_request = {"type":"runInference","dataset":"hcmc_sentinel","experiment":"test","extent":{"xmax":11879090,"xmin":11878790,"ymax":1206070.0000000007,"ymin":1205770.0000000005,"crs":"epsg:3857"},"classes":[{"name":"Class 1","color":"#0000FF","count":0},{"name":"Class 2","color":"#008000","count":0},{"name":"Class 3","color":"#80FF80","count":0},{"name":"Class 4","color":"#806060","count":0}]}

data_loader = DataLoaderCustom("../data/imagery/hcmc_sentinel.tif", [], 500)
raster = data_loader.get_data_from_extent(pred_patch_request["extent"])

upper_left_request = {"type":"correction","dataset":"hcmc_sentinel","experiment":"test","point":{"x":11878791,"y":1206069,"crs":"epsg:3857"},"classes":[{"name":"Class 1","color":"#0000FF","count":0},{"name":"Class 2","color":"#008000","count":0},{"name":"Class 3","color":"#80FF80","count":0},{"name":"Class 4","color":"#806060","count":0}],"value":0}
upper_right_request = {"type":"correction","dataset":"hcmc_sentinel","experiment":"test","point":{"x":11879089,"y":1206070,"crs":"epsg:3857"},"classes":[{"name":"Class 1","color":"#0000FF","count":0},{"name":"Class 2","color":"#008000","count":0},{"name":"Class 3","color":"#80FF80","count":0},{"name":"Class 4","color":"#806060","count":0}],"value":0}
bottom_right_request = {"type":"correction","dataset":"hcmc_sentinel","experiment":"test","point":{"x":11879089,"y":1205771,"crs":"epsg:3857"},"classes":[{"name":"Class 1","color":"#0000FF","count":0},{"name":"Class 2","color":"#008000","count":0},{"name":"Class 3","color":"#80FF80","count":0},{"name":"Class 4","color":"#806060","count":0}],"value":0}

def convert_to_patch_index(request):
    
    lon, lat = request["point"]["x"], request["point"]["y"]
    origin_crs = request["point"]["crs"]

    data_crs, data_transform = raster.crs, raster.transform
    x, y = fiona.transform.transform(origin_crs, data_crs.to_string(), [lon], [lat])
    x = x[0]
    y = y[0]

    dst_col, dst_row = (~data_transform) * (x, y)
    dst_row = int(np.floor(dst_row))
    dst_col = int(np.floor(dst_col))
    
    return dst_col, dst_row


assert convert_to_patch_index(upper_left_request) == (50,50)
assert convert_to_patch_index(upper_right_request) == (79,50)
assert convert_to_patch_index(bottom_right_request) == (80,79)