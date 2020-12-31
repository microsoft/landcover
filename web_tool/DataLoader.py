import os

import numpy as np
from enum import Enum

from urllib.request import urlopen

import fiona
import fiona.transform
import fiona.crs

import shapely
import shapely.geometry

import rasterio
import rasterio.warp
import rasterio.crs
import rasterio.io
import rasterio.mask
import rasterio.transform
import rasterio.merge

import rtree

import mercantile
import utm

import cv2
import pickle

from . import ROOT_DIR
from .DataLoaderAbstract import DataLoader

NAIP_BLOB_ROOT = 'https://naipblobs.blob.core.windows.net/naip'
LC_BLOB_ROOT =  'https://modeloutput.blob.core.windows.net/full-usa-output'


class InMemoryRaster(object):

    def __init__(self, data, crs, transform, bounds):
        """A wrapper around the four pieces of information needed to define a raster datasource.

        Args:
            data (np.ndarray): The data in the raster. This should be formatted as "channels last", i.e. with shape (height, width, number of channels)
            crs (str): The EPSG code describing the coordinate system of this raster (e.g. "epsg:4326")
            transform (affine.Affine): An affine transformation for converting to/from pixel and global coordinates 
            bounds (tuple): A tuple in the format (left, bottom, right, top) / (xmin, ymin, xmax, ymax) describing the boundary of the raster data in the units of `crs`
        """
        assert len(data.shape) == 3
        assert data.shape[2] < data.shape[1] and data.shape[2] < data.shape[0], "We assume that rasters should have larger height/width then number of channels"
        
        self.data = data
        self.crs = crs
        self.transform = transform
        self.bounds = bounds
        self.shape = data.shape
        
        # The following logic can be used to calculate the bounds from the data and transform
        #height, width, _ = dst_img.shape
        #left, top = dst_transform * (0, 0)
        #right, bottom = dst_transform * (width, height)


# ------------------------------------------------------
# Miscellaneous methods
# ------------------------------------------------------

def extent_to_transformed_geom(extent, dst_crs):
    """This function takes an extent in the the format {'xmax': -8547225, 'xmin': -8547525, 'ymax': 4709841, 'ymin': 4709541, 'crs': 'epsg:3857'}
    and converts it into a GeoJSON polygon, transforming it into the coordinate system specificed by dst_crs.
    
    Args:
        extent (dict): A geographic extent formatted as a dictionary with the following keys: xmin, xmax, ymin, ymax, crs
        dst_crs (str): The desired coordinate system of the output GeoJSON polygon as a string (e.g. epsg:4326)

    Returns:
        geom (dict): A GeoJSON polygon
    """
    left, right = extent["xmin"], extent["xmax"]
    top, bottom = extent["ymax"], extent["ymin"]
    src_crs = extent["crs"]

    geom = {
        "type": "Polygon",
        "coordinates": [[(left, top), (right, top), (right, bottom), (left, bottom), (left, top)]]
    }

    if src_crs == dst_crs: # TODO(Caleb): Check whether this comparison makes sense for CRS objects
        return geom
    else:
        return fiona.transform.transform_geom(src_crs, dst_crs, geom)


def warp_data_to_3857(input_raster):
    """Warps an input raster to EPSG:3857

    Args:
        input_raster (InMemoryRaster): An (in memory) raster datasource to warp

    Returns:
        output_raster (InMemoryRaster): The warped version of `input_raster`
    """
    src_height, src_width, num_channels = input_raster.shape
    src_img_tmp = np.rollaxis(input_raster.data.copy(), 2, 0) # convert image to "channels first" format

    x_res, y_res = input_raster.transform[0], -input_raster.transform[4] # the pixel resolution of the raster is given by the affine transformation
    if x_res < 1 and y_res < 1:
        x_res = 1
        y_res = 1

    dst_crs = "epsg:3857"
    dst_bounds = rasterio.warp.transform_bounds(input_raster.crs, dst_crs, *input_raster.bounds)
    dst_transform, width, height = rasterio.warp.calculate_default_transform(
        input_raster.crs,
        dst_crs,
        width=src_width, height=src_height,
        left=input_raster.bounds[0],
        bottom=input_raster.bounds[1],
        right=input_raster.bounds[2],
        top=input_raster.bounds[3],
        resolution=(x_res, y_res) # TODO: we use the resolution of the src_input, while this parameter needs the resolution of the destination. This will break if src_crs units are degrees instead of meters.
    )

    dst_image = np.zeros((num_channels, height, width), np.float32)
    dst_image, dst_transform = rasterio.warp.reproject(
        source=src_img_tmp,
        destination=dst_image,
        src_transform=input_raster.transform,
        src_crs=input_raster.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=rasterio.warp.Resampling.nearest
    )
    dst_image = np.rollaxis(dst_image, 0, 3) # convert image to "channels last" format
    
    return InMemoryRaster(dst_image, dst_crs, dst_transform, dst_bounds)


def crop_data_by_extent(input_raster, extent):
    """Crops the input raster to the boundaries described by `extent`.

    Args:
        input_raster (InMemoryRaster): An (in memory) raster datasource to crop
        extent (dict): A geographic extent formatted as a dictionary with the following keys: xmin, xmax, ymin, ymax, crs

    Returns:
        output_raster: The cropped version of the input raster
    """
    geom = extent_to_transformed_geom(extent, "epsg:3857")
    return crop_data_by_geometry(input_raster, geom, "epsg:3857")


def crop_data_by_geometry(input_raster, geometry, geometry_crs):
    """Crops the input raster by the input geometry (described by `geometry` and `geometry_crs`).

    Args:
        input_raster (InMemoryRaster): An (in memory) raster datasource to crop
        geometry (dict): A polygon in GeoJSON format describing the boundary to crop the input raster to
        geometry_crs (str): The coordinate system of `geometry` as a EPSG code (e.g "epsg:4326")

    Returns:
        output_raster (InMemoryRaster): The cropped version of the input raster
    """
    src_img_tmp = np.rollaxis(input_raster.data.copy(), 2, 0)
    
    if geometry_crs != input_raster.crs:
        geometry = fiona.transform.transform_geom(geometry_crs, input_raster.crs, geometry)

    height, width, num_channels = input_raster.shape
    dummy_src_profile = {
        'driver': 'GTiff', 'dtype': str(input_raster.data.dtype),
        'width': width, 'height': height, 'count': num_channels,
        'crs': input_raster.crs,
        'transform': input_raster.transform
    }
    
    test_f = rasterio.io.MemoryFile()
    with test_f.open(**dummy_src_profile) as test_d:
        test_d.write(src_img_tmp)
    test_f.seek(0)
    with test_f.open() as test_d:
        dst_img, dst_transform = rasterio.mask.mask(test_d, [geometry], crop=True, all_touched=True, pad=False)
    test_f.close()
    
    dst_img = np.rollaxis(dst_img, 0, 3)
    
    height, width, _ = dst_img.shape
    left, top = dst_transform * (0, 0)
    right, bottom = dst_transform * (width, height)

    return InMemoryRaster(dst_img, input_raster.crs, dst_transform, (left, bottom, right, top))


def get_area_from_geometry(geom, src_crs="epsg:4326"):
    """Semi-accurately calculates the area for an input GeoJSON shape in km^2 by reprojecting it into a local UTM coordinate system.

    Args:
        geom (dict): A polygon (or multipolygon) in GeoJSON format
        src_crs (str, optional): The CRS of `geom`. Defaults to "epsg:4326".

    Raises:
        ValueError: This will be thrown if geom isn't formatted correctly, or is not a Polygon or MultiPolygon type

    Returns:
        area (float): The area of `geom` in km^2
    """

    # get one of the coordinates
    try:
        if geom["type"] == "Polygon":
            lon, lat = geom["coordinates"][0][0]
        elif geom["type"] == "MultiPolygon":
            lon, lat = geom["coordinates"][0][0][0]
        else:
            raise ValueError("Polygons and MultiPolygons only")
    except:
        raise ValueError("Input shape is not in the correct format")

    zone_number = utm.latlon_to_zone_number(lat, lon)
    hemisphere = "+north" if lat > 0 else "+south"
    dest_crs = "+proj=utm +zone=%d %s +datum=WGS84 +units=m +no_defs" % (zone_number, hemisphere)
    projected_geom = fiona.transform.transform_geom(src_crs, dest_crs, geom)
    area = shapely.geometry.shape(projected_geom).area / 1000000.0 # we calculate the area in square meters then convert to square kilometers
    return area


# ------------------------------------------------------
# DataLoader for arbitrary GeoTIFFs
# ------------------------------------------------------
class DataLoaderCustom(DataLoader):

    def __init__(self, padding, **kwargs):
        """A `DataLoader` object made for single raster datasources (single .tif files, .vrt files, etc.). This provides functionality for extracting data
        from different shapes and calculating the area of shapes.

        Args:
            padding (float): Amount of padding in terms of units of the CRS of the raster source pointed to by `data_fn`.
            **kwargs: Should contain a "path" key that points to the location of the datasource (e.g. the .tif or .vrt file to use) 
        """
        self._padding = padding
        self.data_fn = kwargs["path"]

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value

    def get_data_from_extent(self, extent):
        with rasterio.open(self.data_fn, "r") as f:
            src_crs = f.crs.to_string()
            transformed_geom = extent_to_transformed_geom(extent, src_crs)
            transformed_geom = shapely.geometry.shape(transformed_geom)

            buffed_geom = transformed_geom.buffer(self.padding)
            buffed_geojson = shapely.geometry.mapping(buffed_geom)

            #buffed = shapely.geometry.mapping(shapely.geometry.box(*buffed_geom.bounds))
            src_image, src_transform = rasterio.mask.mask(f, [buffed_geojson], crop=True, all_touched=True, pad=False) # NOTE: Used to buffer by geom, haven't tested this.

        src_image = np.rollaxis(src_image, 0, 3)
        return InMemoryRaster(src_image, src_crs, src_transform, buffed_geom.bounds)

    def get_data_from_geometry(self, geometry):
        #TODO: Figure out what happens if we call this with a geometry that doesn't intersect the data source.
        f = rasterio.open(self.data_fn, "r")
        src_profile = f.profile
        src_crs = f.crs.to_string()
        transformed_mask_geom = fiona.transform.transform_geom("epsg:4326", src_crs, geometry)
        src_image, src_transform = rasterio.mask.mask(f, [transformed_mask_geom], crop=True, all_touched=True, pad=False)
        f.close()

        src_image = np.rollaxis(src_image, 0, 3)
        return InMemoryRaster(src_image, src_crs, src_transform, shapely.geometry.shape(transformed_mask_geom).bounds)


# ------------------------------------------------------
# DataLoader for US NAIP data and other aligned layers
# ------------------------------------------------------
class NAIPTileIndex(object):
    TILES = None
    
    @staticmethod
    def lookup(geom):
        if NAIPTileIndex.TILES is None:
            assert all([os.path.exists(fn) for fn in [
                "data/tile_index/naip/tile_index.dat",
                "data/tile_index/naip/tile_index.idx",
                "data/tile_index/naip/tiles.p"
            ]]), "You do not have the correct files, did you setup the project correctly"
            NAIPTileIndex.TILES = pickle.load(open("data/tile_index/naip/tiles.p", "rb"))
        return NAIPTileIndex.lookup_naip_tile_by_geom(geom)

    @staticmethod
    def lookup_naip_tile_by_geom(geom):
        minx, miny, maxx, maxy = shapely.geometry.shape(geom).bounds
        geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
        geom = shapely.geometry.shape(geom)

        tile_index = rtree.index.Index("data/tile_index/naip/tile_index")
        intersected_indices = list(tile_index.intersection(geom.bounds))
        for idx in intersected_indices:
            intersected_fn = NAIPTileIndex.TILES[idx][0]
            intersected_geom = NAIPTileIndex.TILES[idx][1]
            if intersected_geom.contains(geom):
                print("Found %d intersections, returning at %s" % (len(intersected_indices), intersected_fn))
                tile_index.close()
                return intersected_fn
        tile_index.close()
        if len(intersected_indices) > 0:
            raise ValueError("Error, there are overlaps with tile index, but no tile completely contains selection")
        else:
            raise ValueError("No tile intersections")


class DataLoaderUSALayer(DataLoader):

    def __init__(self, padding, **kwargs):
        self._padding = padding
        
        # we do this to prime the tile index -- loading the first time can take awhile
        try:
            NAIPTileIndex.lookup(None)
        except:
            pass

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value

    def get_data_from_extent(self, extent):
        query_geom = extent_to_transformed_geom(extent, "epsg:4326")
        naip_fn = NAIPTileIndex.lookup(query_geom)

        with rasterio.open(NAIP_BLOB_ROOT + "/" + naip_fn) as f:
            src_crs = f.crs.to_string()
            transformed_geom = extent_to_transformed_geom(extent, src_crs)
            transformed_geom = shapely.geometry.shape(transformed_geom)

            buffed_geom = transformed_geom.buffer(self.padding)
            buffed_geojson = shapely.geometry.mapping(buffed_geom)

            src_image, src_transform = rasterio.mask.mask(f, [buffed_geojson], crop=True, all_touched=True, pad=False)

        src_image = np.rollaxis(src_image, 0, 3)
        return InMemoryRaster(src_image, src_crs, src_transform, buffed_geom.bounds)

    def get_data_from_geometry(self, geometry):
        naip_fn = NAIPTileIndex.lookup(geometry)

        with rasterio.open(NAIP_BLOB_ROOT + "/" + naip_fn) as f:
            src_profile = f.profile
            src_crs = f.crs.to_string()
            transformed_mask_geom = fiona.transform.transform_geom("epsg:4326", src_crs, geometry)
            src_image, src_transform = rasterio.mask.mask(f, [transformed_mask_geom], crop=True, all_touched=True, pad=False)

        src_image = np.rollaxis(src_image, 0, 3)
        return InMemoryRaster(src_image, src_crs, src_transform, shapely.geometry.shape(transformed_mask_geom).bounds)


# ------------------------------------------------------
# DataLoader for loading RGB data from arbitrary basemaps
# ------------------------------------------------------
class DataLoaderBasemap(DataLoader):

    def __init__(self, padding, **kwargs):
        self._padding = padding
        self.data_url = kwargs["url"]
        self.zoom_level = 17

    @property
    def padding(self):
        return self._padding
    @padding.setter
    def padding(self, value):
        self._padding = value

    def get_image_by_xyz_from_url(self, tile):
        '''NOTE: Here "tile" refers to a mercantile "Tile" object.'''
        req = urlopen(self.data_url.format(
            z=tile.z, y=tile.y, x=tile.x
        ))
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_tile_as_virtual_raster(self, tile):
        '''NOTE: Here "tile" refers to a mercantile "Tile" object.'''
        img = self.get_image_by_xyz_from_url(tile)
        geom = shapely.geometry.shape(mercantile.feature(tile)["geometry"])
        minx, miny, maxx, maxy = geom.bounds
        dst_transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, 256, 256)
        dst_profile = {
            "driver": "GTiff",
            "width": 256,
            "height": 256,
            "transform": dst_transform,
            "crs": "epsg:4326",
            "count": 3,
            "dtype": "uint8"
        }    
        test_f = rasterio.io.MemoryFile()
        with test_f.open(**dst_profile) as test_d:
            test_d.write(img[:,:,0], 1)
            test_d.write(img[:,:,1], 2)
            test_d.write(img[:,:,2], 3)
        test_f.seek(0)
        
        return test_f

    def get_data_from_extent(self, extent):
        transformed_geom = extent_to_transformed_geom(extent, "epsg:4326")
        transformed_geom = shapely.geometry.shape(transformed_geom)
        buffed_geom = transformed_geom.buffer(self.padding)
        
        minx, miny, maxx, maxy = buffed_geom.bounds
        
        virtual_files = [] # this is nutty
        virtual_datasets = []
        for i, tile in enumerate(mercantile.tiles(minx, miny, maxx, maxy, self.zoom_level)):
            f = self.get_tile_as_virtual_raster(tile)
            virtual_files.append(f)
            virtual_datasets.append(f.open())
        out_image, out_transform = rasterio.merge.merge(virtual_datasets, bounds=(minx, miny, maxx, maxy))

        for ds in virtual_datasets:
            ds.close()
        for f in virtual_files:
            f.close()
        
        dst_crs = "epsg:4326"
        dst_profile = {
            "driver": "GTiff",
            "width": out_image.shape[1],
            "height": out_image.shape[0],
            "transform": out_transform,
            "crs": dst_crs,
            "count": 3,
            "dtype": "uint8"
        }
        test_f = rasterio.io.MemoryFile()
        with test_f.open(**dst_profile) as test_d:
            test_d.write(out_image[:,:,0], 1)
            test_d.write(out_image[:,:,1], 2)
            test_d.write(out_image[:,:,2], 3)
        test_f.seek(0)
        test_f.close()

        out_image = np.rollaxis(out_image, 0, 3)
        return InMemoryRaster(out_image, dst_crs, out_transform, buffed_geom.bounds)

    def get_data_from_geometry(self, geometry):
        raise NotImplementedError()



class LCTileIndex(object):
    TILES = None
    
    @staticmethod
    def lookup(geom):
        if LCTileIndex.TILES is None:
            assert all([os.path.exists(fn) for fn in [
                "data/tile_index/lc2019/tile_index.dat",
                "data/tile_index/lc2019/tile_index.idx",
                "data/tile_index/lc2019/tiles.p"
            ]]), "You do not have the correct files, did you setup the project correctly"
            LCTileIndex.TILES = pickle.load(open("data/tile_index/lc2019/tiles.p", "rb"))
        return LCTileIndex.lookup_naip_tile_by_geom(geom)

    @staticmethod
    def lookup_naip_tile_by_geom(geom):
        miny, minx, maxy, maxx = shapely.geometry.shape(geom).bounds
        geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
        geom = shapely.geometry.shape(geom)

        tile_index = rtree.index.Index("data/tile_index/lc2019/tile_index")
        intersected_indices = list(tile_index.intersection(geom.bounds))
        for idx in intersected_indices:
            intersected_fn = LCTileIndex.TILES[idx][0]
            intersected_geom = LCTileIndex.TILES[idx][1]
            if intersected_geom.contains(geom):
                print("Found %d intersections, returning at %s" % (len(intersected_indices), intersected_fn))
                tile_index.close()
                return intersected_fn
        tile_index.close()
        if len(intersected_indices) > 0:
            raise ValueError("Error, there are overlaps with tile index, but no tile completely contains selection")
        else:
            raise ValueError("No tile intersections")

class DataLoaderLCLayer(DataLoader):

    def __init__(self, padding, **kwargs):
        self._padding = padding
        
        # we do this to prime the tile index -- loading the first time can take awhile
        try:
            LCTileIndex.lookup(None)
        except:
            pass

    @property
    def padding(self):
        return self._padding

    @padding.setter
    def padding(self, value):
        self._padding = value

    def get_data_from_extent(self, extent):
        query_geom = extent_to_transformed_geom(extent, "epsg:4326")
        naip_fn = LCTileIndex.lookup(query_geom)

        with rasterio.open(LC_BLOB_ROOT + "/" + naip_fn) as f:
            src_crs = f.crs.to_string()
            transformed_geom = extent_to_transformed_geom(extent, src_crs)
            transformed_geom = shapely.geometry.shape(transformed_geom)

            buffed_geom = transformed_geom.buffer(self.padding)
            buffed_geojson = shapely.geometry.mapping(buffed_geom)

            src_image, src_transform = rasterio.mask.mask(f, [buffed_geojson], crop=True, all_touched=True, pad=False)

        src_image = np.rollaxis(src_image, 0, 3)
        return InMemoryRaster(src_image, src_crs, src_transform, buffed_geom.bounds)

    def get_data_from_geometry(self, geometry):
        naip_fn = LCTileIndex.lookup(geometry)

        with rasterio.open(LC_BLOB_ROOT + "/" + naip_fn) as f:
            src_profile = f.profile
            src_crs = f.crs.to_string()
            transformed_mask_geom = fiona.transform.transform_geom("epsg:4326", src_crs, geometry)
            src_image, src_transform = rasterio.mask.mask(f, [transformed_mask_geom], crop=True, all_touched=True, pad=False)

        src_image = np.rollaxis(src_image, 0, 3)
        return InMemoryRaster(src_image, src_crs, src_transform, shapely.geometry.shape(transformed_mask_geom).bounds)