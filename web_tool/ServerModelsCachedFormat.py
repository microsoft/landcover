import sys, os, time

import numpy as np

import fiona
import fiona.transform
import shapely
import shapely.geometry
import rasterio
import rasterio.mask

import DataLoader

from ServerModelsAbstract import BackendModel

class CachedModel(BackendModel):

    def __init__(self, results_dir):
        self.results_dir = results_dir

    def run(self, naip_data, naip_fn, extent, buffer):
        return self.get_cached_by_extent(naip_fn, extent, buffer)

    def get_cached_by_extent(self, fn, extent, buffer):
        fn = fn.replace("esri-naip/", "full-usa-output/%s/" % (self.results_dir))[:-4] + "_prob.tif"

        f = rasterio.open(fn, "r")
        geom = DataLoader.extent_to_transformed_geom(extent, f.crs["init"])
        pad_rad = buffer # TODO: this might need to be changed for much larger inputs
        buffed_geom = shapely.geometry.shape(geom).buffer(pad_rad)
        minx, miny, maxx, maxy = buffed_geom.bounds
        geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
        out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
        src_crs = f.crs.copy()
        f.close()
        
        dst_crs = {"init": "EPSG:%s" % (extent["spatialReference"]["latestWkid"])}
        dst_transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs,
            dst_crs,
            width=out_image.shape[2], height=out_image.shape[1],
            left=buffed_geom.bounds[0],
            bottom=buffed_geom.bounds[1],
            right=buffed_geom.bounds[2],
            top=buffed_geom.bounds[3],
            resolution=1
        )

        dst_image = np.zeros((out_image.shape[0], height, width), np.uint8)
        rasterio.warp.reproject(
                source=out_image,
                destination=dst_image,
                src_transform=out_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=rasterio.warp.Resampling.nearest
        )
        
        # Calculate the correct padding
        w = extent["xmax"] - extent["xmin"]
        padding = int(np.round((dst_image.shape[1] - w) / 2))
        dst_image = np.rollaxis(dst_image, 0, 3)
        dst_image = dst_image[padding:-padding, padding:-padding, :]
        print(dst_image.shape)

        return dst_image / 255.0

    def retrain(self):
        raise NotImplementedError()

    def add_sample(self, tdst_row, bdst_row, tdst_col, bdst_col, class_idx):
        raise NotImplementedError()

    def reset(self):
        pass
        #raise NotImplementedError()

class CachedMiddleCedarModel(BackendModel):

    def __init__(self, results_dir):

        x_test = []
        y_test = []

        highres_boundary_shapes = []
        highres_fns = [
            "data/highres_prob_predictions_quantized_compressed_5_11_2018_quad1.tif",
            "data/highres_prob_predictions_quantized_compressed_5_11_2018_quad2.tif",
            "data/highres_prob_predictions_quantized_compressed_5_11_2018_quad3.tif"
        ]
        qa_fns = [
            "data/quad1_QAQC1.tif",
            "data/quad2_QAQC1.tif",
            "data/quad3_QAQC1.tif"
        ]
        for fn in highres_fns:
            f = fiona.open(fn[:-4]+".shp","r")
            highres_boundary_shape = shapely.geometry.shape(next(f)["geometry"])
            f.close()
            highres_boundary_shapes.append(highres_boundary_shape)

        if True:
            for i in range(3):
                f = rasterio.open(highres_fns[i],"r")
                highres_raster = f.read().squeeze()
                highres_raster = np.rollaxis(highres_raster, 0, 3)
                print(highres_raster.shape)
                f.close()
                x_test.append(highres_raster.reshape(-1,4) / 255.0)

                f = rasterio.open(qa_fns[i],"r")
                target_raster = f.read().squeeze()
                print(target_raster.shape)
                f.close()
                y_test.append(target_raster.reshape(-1))
            x_test = np.concatenate(x_test, axis=0)
            y_test = np.concatenate(y_test, axis=0)
            mask = y_test != 255
            x_test = x_test[mask]
            y_test = y_test[mask] - 1
            print("Loaded")


        self.results_dir = results_dir

    def run(self, naip_data, naip_fn, extent, buffer):
        return self.get_cached_by_extent(naip_fn, extent, buffer), "Full USA Pre-run %s" % (self.results_dir)

    def get_cached_by_extent(self, fn, extent, buffer):
        fn = fn.replace("esri-naip/", "full-usa-output/%s/" % (self.results_dir))[:-4] + "_prob.tif"

        f = rasterio.open(fn, "r")
        geom = DataLoader.extent_to_transformed_geom(extent, f.crs["init"])
        pad_rad = 15 # TODO: this might need to be changed for much larger inputs
        buffed_geom = shapely.geometry.shape(geom).buffer(pad_rad)
        minx, miny, maxx, maxy = buffed_geom.bounds
        geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
        out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True)
        src_crs = f.crs.copy()
        f.close()
        
        dst_crs = {"init": "EPSG:%s" % (extent["spatialReference"]["latestWkid"])}
        dst_transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs,
            dst_crs,
            width=out_image.shape[2], height=out_image.shape[1],
            left=buffed_geom.bounds[0],
            bottom=buffed_geom.bounds[1],
            right=buffed_geom.bounds[2],
            top=buffed_geom.bounds[3],
            resolution=1
        )

        dst_image = np.zeros((out_image.shape[0], height, width), np.uint8)
        rasterio.warp.reproject(
                source=out_image,
                destination=dst_image,
                src_transform=out_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=rasterio.warp.Resampling.nearest
        )
        
        # Calculate the correct padding
        #w = extent["xmax"] - extent["xmin"]
        #padding = int(np.round((dst_image.shape[1] - w) / 2))

        dst_image = np.rollaxis(dst_image, 0, 3)
        #dst_image = dst_image[padding:-padding, padding:-padding, :]

        return dst_image / 255.0


    def get_cached_by_extent(self, fn, extent, buffer):
    
        geom = DataLoader.extent_to_transformed_geom(extent, "epsg:2794")
        geom = shapely.geometry.shape(geom)
        new_fn = None
        for i, boundary_shape in enumerate(highres_boundary_shapes):
            if boundary_shape.contains(geom):
                new_fn = highres_fns[i]
                break
        
        if new_fn is None:
            print("No intersections")
            new_fn = fn
            new_fn = new_fn.replace("esri-naip/", "full-usa-output/1_3_2019/")[:-4] + "_prob.tif"

        f = rasterio.open(new_fn, "r")
        geom = DataLoader.extent_to_transformed_geom(extent, f.crs["init"])
        pad_rad = 15 # TODO: this might need to be changed for much larger inputs
        buffed_geom = shapely.geometry.shape(geom).buffer(pad_rad)
        minx, miny, maxx, maxy = buffed_geom.bounds
        geom = shapely.geometry.mapping(shapely.geometry.box(minx, miny, maxx, maxy, ccw=True))
        out_image, out_transform = rasterio.mask.mask(f, [geom], crop=True, nodata=-1)
        src_crs = f.crs.copy()
        f.close()
        
        dst_crs = {"init": "EPSG:%s" % (extent["spatialReference"]["latestWkid"])}
        dst_transform, width, height = rasterio.warp.calculate_default_transform(
            src_crs,
            dst_crs,
            width=out_image.shape[2], height=out_image.shape[1],
            left=buffed_geom.bounds[0],
            bottom=buffed_geom.bounds[1],
            right=buffed_geom.bounds[2],
            top=buffed_geom.bounds[3],
            resolution=1
        )

        dst_image = np.zeros((out_image.shape[0], height, width), np.uint8)
        rasterio.warp.reproject(
            source=out_image,
            destination=dst_image,
            src_transform=out_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=rasterio.warp.Resampling.nearest
        )
        
        # Calculate the correct padding
        w = extent["xmax"] - extent["xmin"]
        padding = int(np.round((dst_image.shape[1] - w) / 2))

        dst_image = np.rollaxis(dst_image, 0, 3)
        dst_image = dst_image[padding:-padding, padding:-padding, :]

        return dst_image / 255.0, "highres_prob_predictions_quantized_compressed_5_11_2018"