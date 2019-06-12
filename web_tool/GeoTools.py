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


def get_trimmed(image):
    '''Removes a border of zeros around an image 
    '''
    zeros = image[0,0].copy()

    y = image.shape[0] // 2
    x = 0
    while True:
        if not np.all(image[y,x,:] == zeros):
            break
        x+=1
    left = x

    y = image.shape[0] // 2
    x = image.shape[1]-1
    while True:
        if not np.all(image[y,x,:] == zeros):
            break
        x-=1
    right = x+1

    y = 0
    x = image.shape[1] // 2
    while True:
        if not np.all(image[y,x,:] == zeros):
            break
        y+=1
    top = y

    y = image.shape[0]-1
    x = image.shape[1] // 2
    while True:
        if not np.all(image[y,x,:] == zeros):
            break
        y-=1
    bottom = y+1
    image_clipped = image[top:bottom, left:right,:]
    return image_clipped