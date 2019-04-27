import numpy as np
import rasterio
import fiona
import fiona.transform

def extent_to_transformed_geom(extent, dest_crs="EPSG:4269"):
    left, right = extent["xmin"], extent["xmax"]
    top, bottom = extent["ymax"], extent["ymin"]

    geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }

    src_crs = "EPSG:" + str(extent["spatialReference"]["latestWkid"])

    return fiona.transform.transform_geom(src_crs, dest_crs, geom)