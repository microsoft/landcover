import rasterio
import fiona.transform
import shapely.geometry

INPUT_FN = "../data/imagery/hcmc_sentinel.tif"

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