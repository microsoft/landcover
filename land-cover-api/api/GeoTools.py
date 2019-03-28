import numpy as np
import rasterio
import fiona
from pyproj import Proj, transform

       
patch_size = 500
latest_wkid = 3857
crs = "EPSG:4326"
    
def extent_to_transformed_geom(extent, dest_crs="EPSG:4269"):
    left, right = extent["xmin"], extent["xmax"]
    top, bottom = extent["ymax"], extent["ymin"]

    geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }

    src_crs = "EPSG:" + str(latest_wkid)

    return fiona.transform.transform_geom(src_crs, dest_crs, geom)

def get_geom(lat, lon, src_crs):

    extent = get_extent_values(lat, lon)
    geom = extent_to_transformed_geom(extent)
    crs = src_crs

    return extent, geom

def get_extent_values( lat, lon):
    
    polygon = get_polygon(lat, lon)
    
    topleft = [polygon[0][0], polygon[0][1]]
    topleftProjected = get_projected(topleft[0], topleft[1])
    bottomright = [polygon[2][0], polygon[2][1]]
    bottomrightProjected = get_projected(bottomright[0], bottomright[1])
    
    return {
        "xmax": bottomrightProjected[0],
        "xmin": topleftProjected[0],
        "ymax": topleftProjected[1],
        "ymin": bottomrightProjected[1],
        "spatialreference": {
            "latestwkid": latest_wkid
        }
    }

def get_projected( lat, lon):
    
    src_crs = Proj(init=crs)
    dest_crs = Proj(init='epsg:'+ str(latest_wkid))
    
    x,y= transform(src_crs, dest_crs, lon, lat)
    
    return x, y   

def get_unprojected( x, y):
    
    src_crs = Proj(init='epsg:' + str(latest_wkid), preserve_units = True)
    dest_crs = Proj(init=crs)

    lat,lon = transform(src_crs, dest_crs, x, y)
    
    return lon, lat

def get_polygon( lat, lon):
    
    latlonProjected = get_projected(lat, lon)
    x = latlonProjected[0]
    y = latlonProjected[1]
   
    top = y + (patch_size/2)
    bottom = y - patch_size/2
    left = x - patch_size/2
    right = x + patch_size/2

    top = int(round(top))
    bottom = int(round(bottom))
    left = int(round(left))
    right = int(round(right))
    
    topleft = get_unprojected(left, top)
    bottomright = get_unprojected(right, bottom)
                
    return( [
        [topleft[0], topleft[1]],
        [topleft[0], bottomright[1]] ,
        [bottomright[0], bottomright[1]] ,
        [bottomright[0], topleft[1]]
    ] )

    