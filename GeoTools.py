import numpy as np
import rasterio
import fiona

def extent_to_transformed_geom(extent, dest_crs="EPSG:4269"):
    left, right = extent["xmin"], extent["xmax"]
    top, bottom = extent["ymax"], extent["ymin"]

    geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }

    # The map navigator uses EPSG:3857 and Caleb's indices use EPSG:4269
    return fiona.transform.transform_geom("EPSG:3857", dest_crs, geom)