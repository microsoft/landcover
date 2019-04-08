import random
from pyproj import Proj, transform
import enum

class Options(enum.Enum): 
    tile = 1
    classify = 2
    tilebyExtent = 3
    classifybyExtent = 4

def generate_random_USA_coordinate():
     # US points
    lat_max = 40.
    lat_min = 30.
    lon_max = -90.
    lon_min = -100.

    lat = get_random_value(lat_min, lat_max)
    lon = get_random_value(lon_min, lon_max)

    return lat, lon

def get_random_value(min, max):
    val =  round(random.uniform(min, max), 6)
    return val

def get_projected(lat, lon):
    
    P3857 = Proj(init='epsg:3857', preserve_units = True)
    P4326 = Proj(init='epsg:4326')
    
    x,y= transform(P4326, P3857, lon, lat)
    
    return x, y   

def get_unprojected(x, y):
    
    P3857 = Proj(init='epsg:3857', preserve_units = True)
    P4326 = Proj(init='epsg:4326')

    lat,lon = transform(P3857, P4326, x, y)
    
    return lon, lat

def get_polygon(lat, lon, patch_size):
    
    latlonProjected = get_projected(lat, lon)
    x = latlonProjected[0]
    y = latlonProjected[1]
    
    top = y + patch_size/2
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

def get_extent_values(lat, lon, patch_size):
    
    polygon = get_polygon(lat, lon, patch_size)
    topleft = [polygon[0][0], polygon[0][1]]
    topleftProjected = get_projected(topleft[0], topleft[1])
    bottomright = [polygon[2][0], polygon[2][1]]
    bottomrightProjected = get_projected(bottomright[0], bottomright[1])
    
    xmax =  bottomrightProjected[0]
    xmin =  topleftProjected[0]
    ymax =  topleftProjected[1]
    ymin =  bottomrightProjected[1]
    
    return {"xmax": xmax, "xmin" : xmin, "ymax" :ymax, "ymin": ymin }


def get_tile_input_json(choice, random=True):
    if(random):
        lat, lon = generate_random_USA_coordinate()
        patch_size = int(round(random.uniform(50, 700)))
    else:
        lat = 47.64432195
        lon = -122.136437243892
        patch_size = 500

    input_json = {}

    if(choice == Options.tile.value):
        input_json = {"lat": lat, "lon": lon, "patchSize" : patch_size, "latestWkid": 3857}
    elif(choice == Options.classify.value):
        input_json = {"lat": lat, "lon": lon, "patchSize" : patch_size, "latestWkid": 3857, 
                      "weights" : [0.25,0.25,0.25,0.25]}
    elif(choice == Options.tilebyExtent.value):
        val = get_extent_values(lat, lon, patch_size)
        input_json = {"extent": {"xmin":val["xmin"], "xmax": val["xmax"], "ymin":val["ymin"], "ymax":val["ymax"],
        "spatialReference":{"latestWkid":3857}}}
    elif(choice == Options.classifybyExtent.value):
        val = get_extent_values(lat, lon, patch_size)
        input_json = {"extent": {"xmin":val["xmin"], "xmax": val["xmax"], "ymin":val["ymin"], "ymax":val["ymax"],
                      "spatialReference":{"latestWkid":3857}}, "weights":[0.25,0.25,0.25,0.25]}
                      
    return input_json

if __name__ == '__main__':
    print("1. Generate input data for /tile method")
    print("2. Generate input data for /classify method")
    print("3. Generate input data for /tile_by_extent method")
    print("4. Generate input data for /classify_by_extent method")

    choice = input("\nEnter the number of your choice from the list above: ")

    while choice not in ["1","2","3","4"]:
        print("In valid choice")
        choice = input("\nEnter the number of your choice from the list above: ")
    
    input_data = get_tile_input_json(int(choice), random=True)
    print(input_data)



