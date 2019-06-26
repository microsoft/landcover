#!/usr/bin/env python

"""
Utilities for working with Open Street Map
"""
from shapely.ops import cascaded_union
import cv2
import json
import numpy as np
import os
import requests
import shapely


def write_geojson(polygon, fname="output", query=None, endpoint=None):
    """
    GeoJSONs associated with a Bounding Box

    This downloads shapefiles from OpenStreetMap's overpass api and converts
    them to geojson, using the npm module ostmtogeojson (which must be
    installed on the system before calling this function).

    :param polygon: A list of [lat, lon] lists giving vertices of the polygon
      within which to return OSM data.
    :param fname: The path / name to write the resulting geojson to.
    :param query: The prefix to the overpass API query. Defaults to
      way['waterway'='river'], though there are other geographical features
      that might be interesting to derive
    :param endpoint: The base of the endpoint containing the geometries.
    :return None, but writes fname.geojson as a side-effect.

    Example
    >>> polygon = json.load(open("../../data/specification/region_01.json", "r"))["coordinates"][0]
    >>> write_geojson(polygon)
    """
    polygon = " ".join([" ".join([str(t) for t in s[::-1]]) for s in polygon])

    if not endpoint:
        endpoint = "https://overpass-api.de/api/interpreter?data"

    if not query:
        query_elems = [
            "way['waterway'='river']",
            "way['waterway'='stream']",
            "relation['waterway'='riverbank']",
            "way['natural'='water']",
        ]
        query = "(poly: '{}');".format(polygon).join(query_elems)

    query_final = "[out:json];({}(poly: '{}'););(._;>;);out;".format(query, polygon)
    write_geojson_(query_final, fname)


def write_geojson_(query_final, fname):
    """
    Internal fun to write a overpass result

    This writes the result of a call to the overpass API, once the query has
    been finalized.
    """
    result = requests.get("http://overpass-api.de/api/interpreter?data={}".format(query_final))
    with open("tmp.json", "w") as f:
        json.dump(result.json(), f)

    os.system("osmtogeojson tmp.json > {}.geojson".format(fname))
    os.remove("tmp.json")


def make_image(geojson, bounds_mat, im_size=(512, 512), buffer_size=0.0001):
    """
    Get the Image associated with a geojson

    This creates a binary image mask associated with the polygons in a geojson,
    restricted to some bounding box. At the moment, this bounding box must be
    axis aligned.

    :param geojson: A geojson giving all the line, point, and polygon features.
    :param bounds_mat: A numpy array giving the coordinates surrounding the
      features of interest.
    :param im_size: A tuple giving the size of the output image.
    :param buffer_size: For line features, how wide should we pad the lines in
      the output image?

    Example
    --------
    >>> bounds = json.load(open("image_bounds.json")) # defined according to the visual asset below
    >>> write_geojson(np.array(bounds["coordinates"]), "query_result")
    >>> geojson = json.load(open("query_result.geojson", "r"))
    >>> satellite_image = cv2.imread("/Users/t-anorti/Desktop/red-cross-expers/exper/planet_download/PSScene3Band_20170901_032915_1004_visual")
    >>> image = make_image(geojson, np.array(bounds["coordinates"]), satellite_image.shape[:2])
    >>> plt.imshow(satellite_image, alpha = 0.9)
    >>> plt.imshow(image, alpha = 0.1)
    """
    im_size = im_size[:2]
    img_mask = np.zeros(im_size)
    int_coords = lambda x: np.array(x).round().astype(np.int32)

    # buffer geoms if they're not already polygons
    exteriors, geoms = [], []
    for feature in geojson["features"]:
        geom = shapely.geometry.shape(feature["geometry"])
        if geom.geom_type != "Polygon":
            geom = geom.buffer(buffer_size)
        geoms.append(geom)

    # get images for each geom
    for geom in geoms:
        if geom.is_empty:
            continue
        exteriors = np.array(geom.exterior.coords)
        exteriors = transform_scale(exteriors, bounds_mat, im_size)
        img_mask += cv2.fillPoly(np.zeros(im_size), [int_coords(exteriors)], 1)

    return (img_mask > 0).astype(float)


def transform_scale(coords, bounds0, im_size):
    """
    Transform lat / longs into pixel coordinates

    This maps between original geographic coordinates and pixel coordinates,
    according to the way cv2.imread reads in tiffs. We have to make some
    assumptions about the cardinal directions of the endpoints in the image,
    unless we can enforce some standard on the bounds input.
    """
    bounds = np.vstack([np.min(bounds0, axis=0), np.max(bounds0, axis=0)])

    scaled_coords = np.array([
        (coords[:, 0] - bounds[0, 0]),
        (coords[:, 1] - bounds[0, 1])
    ]).T
    scaled_coords /= np.ptp(bounds, axis=0)
    scaled_coords[:, 1] = im_size[0] * (1 - scaled_coords[:, 1])
    scaled_coords[:, 0] *= im_size[1]
    return scaled_coords


def load_json(query_skeleton, query):
    query_final = query_skeleton.format(query)
    write_geojson_(query_final, "tmp")
    with open("tmp.geojson", "r") as f:
        result = json.load(f)

    os.remove("tmp.geojson")
    return result


def labels_for_patch(shape, coord):
    """
    Helper to get water and road masks

    The strategy is to download the openstreetmap features for a particular
    patch (specified by coord) and then define the binary mask associated with
    it.

    Note that the geojsons output by this are not restricted to the bounding
    box specified by coord, they are the raw geojsons from overpass, which
    includes anything intersecting with the bounding box.


    :param shape: A tuple giving the x and y dimension of the output patch masks.
    :param coord: A numpy array giving the corners of the bounding box (each
      row is one corners) in lat / longs.
    """
    query_skeleton = "[output:json];({});(._;>;); out;"
    masks, geojsons = {}, {}

    # labels for roads
    poly = " ".join([" ".join([str(t) for t in s[::-1]]) for s in coord])
    road_query = "way['highway'](poly: '{}');".format(poly)
    geojsons["road"] = load_json(query_skeleton, road_query)
    masks["road"] = make_image(geojsons["road"], coord, shape, buffer_size=1e-5)

    # labels for water
    water_query = """
    way[waterway](poly: '{p}');
    relation[waterway](poly: '{p}');
    way[natural='water'](poly: '{p}');
    """.format(p=poly)
    geojsons["water"] = load_json(query_skeleton, water_query)
    masks["water"] = make_image(geojsons["water"], coord, shape, buffer_size=1e-5)

    return geojsons, masks