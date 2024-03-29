import sys
import rasterio
import fiona.transform
import shapely.geometry

INPUT_FN = sys.argv[1]

with rasterio.open(INPUT_FN) as f:
    crs = f.crs.to_string()
    
    shape = shapely.geometry.box(*f.bounds)
    geom = shapely.geometry.mapping(shape)
    
    geom = fiona.transform.transform_geom(crs, "epsg:4326", geom)
    
    top_lng, top_lat = geom["coordinates"][0][0]
    bot_lng, bot_lat = geom["coordinates"][0][2]
    
    print([
        [top_lat,top_lng],
        [bot_lat,bot_lng],
    ])

    centroid_geom = shapely.geometry.mapping(shape.centroid)
    centroid_geom = fiona.transform.transform_geom(crs, "epsg:4326", centroid_geom)
    print(centroid_geom["coordinates"][1], centroid_geom["coordinates"][0])
