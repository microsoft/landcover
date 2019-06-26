#!/usr/bin/env python

"""
Utiltiies for obtaining satellite data
"""
from requests.auth import HTTPBasicAuth
from retrying import retry
import dateutil.parser
import os
import os.path
import re
import requests
import time
import xml.etree.ElementTree

def search_pages(geometry, start=None, end=None, api_key=None):
    """
    Links to / Metadata for raw Planet API Images

    :param geometry: A geojson geometry giving a polygon within which to
      restrict the search for images.
    :param start: The datetime (as a string) giving the earliest acquisition
      time for any of the images we want.
    :param end: The datetime (as a string) giving the latest acquisition time
      for any of the images we want.
    :param api_key: The Planet API key that authorizes our request.
    :return result: A json with three keys, '_links', 'features', and 'type'.
      _links gives the paths that we'll need to activate in order to download
      images. 'features' gives us metadata describing each image (e.g., it's
      bounding box, and the satellite it was taken from).

    >>> geo_json_geometry = {
    >>>     "type": "Polygon",
    >>>     "coordinates": [
    >>>         [
    >>>             [-118.38317871093749, 34.05379721731628],
    >>>             [-118.21495056152342, 34.05379721731628],
    >>>             [-118.21495056152342, 34.11748941036342],
    >>>             [-118.38317871093749, 34.11748941036342],
    >>>             [-118.38317871093749, 34.05379721731628]
    >>>         ]
    >>>         ]
    >>>     }
    >>> result = search_pages(geo_json_geometry, api_key=api_key)
    """
    # fill in some defaults
    if not start:
        start = "2016-07-01T00:00:00.000Z"
    if not end:
        end = "2016-08-01T00:00:00.000Z"

    # define the filters
    geometry_filter = {
        "type": "GeometryFilter",
        "field_name": "geometry",
        "config": geometry
    }

    date_range_filter = {
        "type": "DateRangeFilter",
        "field_name": "acquired",
        "config": {"gte": start, "lte": end}
    }

    complete_filter = {
        "type": "AndFilter",
        "config": [geometry_filter, date_range_filter]
    }

    # make the endpoint request
    endpoint_request = {
        "item_types": ["PSOrthoTile", "REOrthoTile", "PSScene3Band",
                      "PSScene4Band", "REScene", "Landsat8L1G", "Sentinel2L1C"],
        "filter": complete_filter
    }

    results = [
        requests.post(
            "https://api.planet.com/data/v1/quick-search",
            auth=HTTPBasicAuth(api_key, ""),
            json=endpoint_request
        ).json()
    ]

    while results[-1]["_links"].get("_next"):
        print("activating {}".format(results[-1]["_links"].get("_next")))
        results.append(
            requests.get(
                results[-1]["_links"]["_next"],
                auth=HTTPBasicAuth(api_key, "")
            ).json()
        )

    return results


def throw_err(status_code):
    if status_code != 200:
        raise Exception("Received response {}".format(status_code))


@retry(stop_max_attempt_number=30)
def activate_item(item_type, item_id, api_key, keep_types=None):
    """
    Activate a Planet Asset

    :param item_type: The item type associated with an asset. This is available
      in the properties["item_type:"] field of a call to the Planet search API.
    :param item_id: The Planet ID used to specify an asset.
    :param api_key: The Planet API key that authorizes our request.
    :param keep_types: Which types of assets should we be downloading? By
      default, look for the basic XML metadata and the visual image.E
    :return: None, but downloads assets to the out directory.

    Example
    -------
    >>> geo_json_geometry = {
    >>>     "type": "Polygon",
    >>>     "coordinates": [
    >>>         [
    >>>             [-118.38317871093749, 34.05379721731628],
    >>>             [-118.21495056152342, 34.05379721731628],
    >>>             [-118.21495056152342, 34.11748941036342],
    >>>             [-118.38317871093749, 34.11748941036342],
    >>>             [-118.38317871093749, 34.05379721731628]
    >>>         ]
    >>>         ]
    >>>     }
    >>> result = search_pages(geo_json_geometry)
    >>> asset_url = result[0]["features"][0]["_links"]["assets"]
    >>> activate_url(asset_url, $PL_API_KEY)
    """
    # request an item
    print("trying " + item_id)
    item_url = "https://api.planet.com/data/v1/item-types/{}/items/{}/assets/".format(
        item_type,
        item_id
    )

    # raise an exception to trigger the retry
    session = requests.Session()
    session.auth = (api_key, "")
    item = session.get(item_url)
    throw_err(item.status_code)

    if not keep_types:
        keep_types = ["analytic_xml", "analytic"]

    inter = set(item.json().keys()).intersection(keep_types)
    if len(inter) != len(keep_types):
        return

    # request activation
    for asset_type in keep_types:
        activation_url = item.json()[asset_type]["_links"]["activate"]
        session.post(activation_url)
        time.sleep(20)
        item = session.get(item_url)
        throw_err(item.status_code)
        print(item.status_code)
        print(item.json()[asset_type])
        if item.json()[asset_type]["status"] != "active":
            raise Exception("Item still inactive.")

    return item_id


def download_item(item_type, item_id, api_key, out_dir=".", keep_types=None):
    """
    Download Planet asset

    This downloads a planet asset after we've already activated its URL. We're
    following the discussion here:
    https://developers.planet.com/docs/quickstart/downloading-imagery/

    :param item_type: The item type associated with an asset. This is available
      in the properties["item_type:"] field of a call to the Planet search API.
    :param item_id: The Planet ID used to specify an asset.
    :param api_key: The Planet API key that authorizes our request.
    :param out_dir: The directory to which to save the results.
    :return: None, but downloads assets to the out directory.
    """
    session = requests.Session()
    session.auth = (api_key, '')
    asset_url = "https://api.planet.com/data/v1/item-types/{}/items/{}/assets/".format(
        item_type,
        item_id
    )
    item = session.get(asset_url)

    if not keep_types:
        keep_types = ["analytic_xml", "analytic"]
    asset_exts = {"analytic_xml": "xml", "analytic": "tiff"}

    for asset_type in keep_types:
        if not item.json()[asset_type].get("location"):
            raise Exception("Item not active")
        download_url = item.json()[asset_type]["location"]

        # only download if we don't already have the asset
        output_base = "{}_{}.{}".format(item_type, item_id, asset_exts[asset_type])
        output_path = os.path.join(out_dir, output_base)
        if not os.path.exists(output_path):
            curl_cmd = "curl -L -H 'Authorization: api-key {}' '{}' > {}".format(
                api_key,
                download_url,
                output_path
            )
            os.system(curl_cmd)


def parallel_downloads(assets, api_key, **kwargs):
    """
    Activate and Download in Parallel

    This activates and downloads visual and xml assets from the Planet API,
    according to the discussion here:
    https://developers.planet.com/docs/quickstart/best-practices-large-aois/

    :param assets: A list of dictionaries giving the item_types and item_ids
      that we want to downloads.
    :param api_key: The key that allows access to the google maps API.
    :return None, but downloads satellite images and their XML metadata when
      they are both available.

    >>> assets = [
       {"item_type": "PSScene3Band", "id": "20180605_033916_1033"},
       {"item_type": "REOrthoTile", "id": "20180315_042412_4646623_RapidEye-1"}
    ]
    >>> parallel_downloads(assets, api_key, out_dir="/Users/t-anorti/Desktop/")
    """
    for elem in assets:
        cur_id = activate_item(elem["item_type"], elem["id"], api_key)
        if cur_id:
            download_item(elem["item_type"], elem["id"], api_key, **kwargs)


def planet_xml(fname, prefix="ps"):
    """
    Parse the XML returned from Planet

    This extracts relevant metadata features from the analytic XMLs provided by
    the Planet API.

    :param fname: The full path to the xml file containing planet image
      metadata that we want to parse.
    :param prefix: The prefix for the namespaces specific to the item type that
      we have downloaded data for. ps for planetscope and re for rapideye.
    :return result: A nested dictionary containing relevant metadata.
    """
    # get data and define namespaces
    root = xml.etree.ElementTree.parse(fname).getroot()
    ns = {
        "eop": "http://earth.esa.int/eop",
        "gml": "http://www.opengis.net/gml",
        "opt": "http://earth.esa.int/opt",
        "ps": "http://schemas.planet.com/ps/v1/planet_product_metadata_geocorrected_level",
        "re": "http://schemas.planet.com/products/productMetadataGeocorrected"
    }

    def xpath(path, return_type=str):
        """
        Helper to extract XML values via xpath selectors
        """
        path = path.replace("{prefix}", prefix)
        result = root.find(path, ns).text
        if return_type == float:
            result = float(result)
        elif return_type == int:
            result = int(result)
        return result

    # logic to convert xml to dictionary
    result = {
        "id": xpath(".//eop:identifier"),
        "station": xpath(".//eop:acquisitionStation"),
        "time": xpath(".//{prefix}:Acquisition/{prefix}:acquisitionDateTime"),
        "dimension": [
            xpath(".//{prefix}:numRows", int),
            xpath(".//{prefix}:numColumns", int),
            xpath(".//{prefix}:numBands", int)
        ],
        "cloud_cover": xpath(".//opt:cloudCoverPercentage", float),
        "satellite": {
            "name": xpath(".//eop:Platform/eop:shortName"),
            "ID": xpath(".//eop:Platform/eop:serialIdentifier"),
            "orbit_type": xpath(".//eop:orbitType"),
            "instrument": xpath(".//eop:Instrument/eop:shortName"),
            "resolution": xpath(".//eop:resolution", float)
        },
        "acquisition_params": {
            "direction": xpath(".//eop:orbitDirection"),
            "angle": xpath(".//eop:incidenceAngle", float),
            "azimuth_angle": xpath(".//{prefix}:azimuthAngle", float),
            "spacecraft_angle": xpath(".//{prefix}:spaceCraftViewAngle", float),
            "illumination_azimuth": xpath(".//opt:illuminationAzimuthAngle", float),
            "illumination_elevation": xpath(".//opt:illuminationElevationAngle", float)
        },
        "region": {
            "polygon": xpath(".//gml:coordinates"),
            "center": xpath(".//gml:target//gml:pos"),
            "corners": {
                "top_left": [
                    xpath(".//{prefix}:topLeft/{prefix}:longitude", float),
                    xpath(".//{prefix}:topLeft/{prefix}:latitude", float)
                ],
                "top_right": [
                    xpath(".//{prefix}:topRight/{prefix}:longitude", float),
                    xpath(".//{prefix}:topRight/{prefix}:latitude", float)
                ],
                "bottom_left": [
                    xpath(".//{prefix}:bottomLeft/{prefix}:longitude", float),
                    xpath(".//{prefix}:bottomLeft/{prefix}:latitude", float)
                ],
                "bottom_right": [
                    xpath(".//{prefix}:bottomRight/{prefix}:longitude", float),
                    xpath(".//{prefix}:bottomRight/{prefix}:latitude", float)
                ]
            }
        }
    }

    # final transformations, and return
    result["time"] = dateutil.parser.parse(result["time"])
    result["region"]["center"] = [float(x) for x in result["region"]["center"].split(" ")]
    return result


def google_defaults():
    """
    Default options for the google maps static API
    """
    return {
        "zoom": "18",
        "size": "629x629",
        "maptype": "satellite"
    }


def google_map(center, fname, api_key, **kwargs):
    """
    Get Satellite Image from Google Maps

    :param center: The center of the image to return, as at [latitude,
      longitude] list of floats.
    :param fname: The name of the output file that we will write.
    :param api_key: The key that allows access to the google maps API.
    :param **kwargs: Any parameters to pass to the api endpoint query string.
    :return Nothing, but writes fname.png to file with the required satellite
      image.

    >>> center = [16.3108926, 95.1613735]
    >>> google_map(center, "test_image")
    """
    # fill in query parameters
    opts = google_defaults()
    opts["key"] = api_key
    for k, v in kwargs.items():
        opts[k] = str(v)

    # convert query to a string
    opts_str = ""
    for k, v in opts.items():
        opts_str += "&" + k + "=" + v

    # make the request
    center_str = ",".join([str(s) for s in center])
    query = "https://maps.googleapis.com/maps/api/staticmap?center={}{}".format(center_str, opts_str)
    os.system("curl '{}' >> {}.png".format(query, fname))