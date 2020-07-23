'''This script takes a raster file, number of rows, and number of columns as input, then creates a set of polygons that tile the extent of the input raster as output.

The polygons are saved as GeoJSON and warped to EPSG:4326.
'''
import os
from collections import OrderedDict

import numpy as np

import rasterio
import fiona
import fiona.crs
import shapely.geometry

input_fn = "../data/imagery/m_3807537_ne_18_1_20170611.tif"
output_fn = "m_3807537_ne_18_1_20170611_fishnet.geojson"

num_rows = 5
num_cols = 5

with rasterio.open(input_fn) as f:
    bounds = f.bounds
    crs = f.crs

top = bounds.top
left = bounds.left

tile_width = bounds.right - bounds.left
tile_height = bounds.top - bounds.bottom

patch_width = tile_width / num_cols
patch_height = tile_height / num_rows

bounding_box_schema = {
    'geometry': 'Polygon',
    'properties': OrderedDict([
        ('id', 'int')
    ])
}

with fiona.open("tmp.geojson", "w", driver="GeoJSON", crs=crs, schema=bounding_box_schema) as f:
    
    idx = 0
    for y_idx in range(num_rows):
        for x_idx in range(num_cols):
    
            t_top = top - (y_idx*patch_height)
            t_bottom = top - (y_idx*patch_height) - patch_height
            t_left = left + (x_idx*patch_width)
            t_right = left + (x_idx*patch_width) + patch_width
    
            shape = shapely.geometry.box(minx=t_left, miny=t_bottom, maxx=t_right, maxy=t_top)
            geometry = shapely.geometry.mapping(shape)
            row = {
                "geometry": geometry,
                "properties": OrderedDict([
                    ("id", idx)
                ])
            }
            f.write(row)
            idx += 1
            
os.system(f"ogr2ogr -t_srs epsg:4326 -f GeoJSON {output_fn} tmp.geojson")
os.remove("tmp.geojson")