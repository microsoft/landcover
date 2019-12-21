import os

import utm
import fiona
import fiona.transform
import shapely
import shapely.geometry
from enum import Enum

from web_tool import ROOT_DIR

from DataLoader import DataLoaderCustom, DataLoaderUSALayer, DataLoaderBasemap

class DatasetTypes(Enum):
    CUSTOM = 1
    USA_LAYER = 2
    BASEMAP = 3

DATASETS = {} # This dictionary should be the only thing imported from other files
DATASET_DEFINITIONS = {
    "esri_world_imagery": {
        "name": "ESRI World Imagery",
        "imagery_metadata": "ESRI World Imagery",
        "data_layer_type": DatasetTypes.BASEMAP,
        "data_url": 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        "data_padding": 0.0005,
        "leafletTileLayer": {
            "url": 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            "args": {
                "minZoom": 4,
                "maxZoom": 20,
                "attribution": 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            }
        },
        "shape_layers": None,
        "location": {
            "center": [38, -88],
            "initialZoom": 4,
            "bounds": None
        }
    },
    "esri_world_imagery_naip": {
        "name": "ESRI World Imagery",
        "imagery_metadata": "ESRI World Imagery",
        "data_layer_type": DatasetTypes.USA_LAYER,
        "data_padding": 20,
        "leafletTileLayer": {
            "url": 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            "args": {
                "minZoom": 4,
                "maxZoom": 20,
                "attribution": 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            }
        },
        "shape_layers": None,
        "location": {
            "center": [38, -88],
            "initialZoom": 4,
            "bounds": None
        }
    },
    "user_study_5": {
        "name": "User Study Area 5",
        "imagery_metadata": "NAIP Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/user_study_5.tif",
        "data_padding": 20,
        "leafletTileLayer": {
            "url": 'tiles/user_study_5/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 13,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "Area boundary", "shapes_fn": "shapes/user_study_5_outline.geojson", "zone_name_key": None}
        ],
        "location": {
            "center": [42.448269618302362, -75.110429001207137],
            "initialZoom": 13,
            "bounds": None
        }
    },
    "leuser": {
        "name": "Sumatra",
        "imagery_metadata": "Planet",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/merged_4m_rgb.tif",
        "data_padding": 500,
        "leafletTileLayer": {
            "url": 'tiles/leuser/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 8,
                "maxNativeZoom": 16,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "Btoru", "shapes_fn": "shapes/btoru_aoi_extent.geojson", "zone_name_key": "MINX"},
            {"name": "Atimur", "shapes_fn": "shapes/atimur_aoi_extent.geojson", "zone_name_key": "MINX"},
        ],
        "location": {
            "center": [4.1828740090001588, 97.418702680771901],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "yangon_sentinel": {
        "name": "Yangon, Myanmar",
        "imagery_metadata": "Sentinel Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/yangon.tif",
        "data_padding": 1100,
        "leafletTileLayer": {
            "url": 'tiles/yangon/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 10,
                "maxNativeZoom": 16,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "States", "shapes_fn": "shapes/yangon_sentinel_admin_1_clipped.geojson", "zone_name_key": "ST"},
            {"name": "Districts", "shapes_fn": "shapes/yangon_sentinel_admin_2_clipped.geojson", "zone_name_key": "DT"},
            {"name": "Townships", "shapes_fn": "shapes/yangon_sentinel_admin_3_clipped.geojson", "zone_name_key": "TS"},
            {"name": "Wards", "shapes_fn": "shapes/yangon_sentinel_admin_4_clipped.geojson", "zone_name_key": "Ward"}
        ],
        "location": {
            "center": [16.66177, 96.326427],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "hcmc_sentinel": {
        "name": "Hồ Chí Minh City, Vietnam",
        "imagery_metadata": "Sentinel Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/hcmc_sentinel.tif",
        "data_padding": 1100,
        "leafletTileLayer": {
            "url": 'tiles/hcmc_sentinel_tiles/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 10,
                "maxNativeZoom": 16,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "Provinces", "shapes_fn": "shapes/hcmc_sentinel_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Districts", "shapes_fn": "shapes/hcmc_sentinel_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Wards", "shapes_fn": "shapes/hcmc_sentinel_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "location": {
            "center": [10.682, 106.752],
            "initialZoom": 11,
            "bounds": None
        }
    },
    "hcmc_sentinel_2017": {
        "name": "Hồ Chí Minh City, Vietnam",
        "imagery_metadata": "Sentinel Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/hcmc_sentinel_2017_01_08.tif",
        "data_padding": 1100,
        "leafletTileLayer": {
            "url": 'tiles/hcmc_sentinel_2017_01_08/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 10,
                "maxNativeZoom": 16,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "Provinces", "shapes_fn": "shapes/hcmc_sentinel_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Districts", "shapes_fn": "shapes/hcmc_sentinel_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Wards", "shapes_fn": "shapes/hcmc_sentinel_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "location": {
            "center": [10.682, 106.752],
            "initialZoom": 11,
            "bounds": None
        }
    },
    "hcmc_spot_2019": {
        "name": "Hồ Chí Minh City, Vietnam",
        "imagery_metadata": "SPOT Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        #"data_fn": "tiles/spot6_2019_02_10.tif",
        "data_fn": "tiles/nhabe_spot_stacked_2019.tif",
        "data_padding": 10,
        "leafletTileLayer": {
            "url": 'tiles/spot6_2019_02_10/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 13,
                "maxNativeZoom": 16,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "Provinces", "shapes_fn": "shapes/hcmc_sentinel_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Districts", "shapes_fn": "shapes/hcmc_sentinel_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Wards", "shapes_fn": "shapes/hcmc_sentinel_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "location": {
            "center": [10.682, 106.752],
            "initialZoom": 13,
            "bounds": None
        }
    },
    "hcmc_spot_2017": {
        "name": "Hồ Chí Minh City, Vietnam",
        "imagery_metadata": "SPOT Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        #"data_fn": "tiles/spot6_2017_02_12.tif",
        "data_fn": "tiles/nhabe_spot_stacked_2017.tif",
        "data_padding": 10,
        "leafletTileLayer": {
            "url": 'tiles/spot6_2017_02_12/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 13,
                "maxNativeZoom": 16,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "Provinces", "shapes_fn": "shapes/hcmc_sentinel_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Districts", "shapes_fn": "shapes/hcmc_sentinel_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Wards", "shapes_fn": "shapes/hcmc_sentinel_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "location": {
            "center": [10.682, 106.752],
            "initialZoom": 13,
            "bounds": None
        }
    },
    "hcmc_pleadies_2019": {
        "name": "Hồ Chí Minh City, Vietnam",
        "imagery_metadata": "Pleadies Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/pleadies_2019_02_10.tif",
        "data_padding": 0,
        "leafletTileLayer": {
            "url": 'tiles/pleadies_2019_02_10/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 13,
                "maxNativeZoom": 16,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "Provinces", "shapes_fn": "shapes/hcmc_sentinel_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Districts", "shapes_fn": "shapes/hcmc_sentinel_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Wards", "shapes_fn": "shapes/hcmc_sentinel_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "location": {
            "center": [10.682, 106.752],
            "initialZoom": 13,
            "bounds": None
        }
    },
    "yangon_lidar": {
        "name": "Yangon, Myanmar",
        "imagery_metadata": "LiDAR Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/yangon_lidar.tif",
        "data_padding": 20,
        "leafletTileLayer": {
            "url": 'tiles/yangon_lidar/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 10,
                "maxNativeZoom": 20,
                "maxZoom": 21,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "States", "shapes_fn": "shapes/yangon_lidar_admin_1_clipped.geojson", "zone_name_key": "ST"},
            {"name": "Districts", "shapes_fn": "shapes/yangon_lidar_admin_2_clipped.geojson", "zone_name_key": "DT"},
            {"name": "Townships", "shapes_fn": "shapes/yangon_lidar_admin_3_clipped.geojson", "zone_name_key": "TS"},
            {"name": "Wards", "shapes_fn": "shapes/yangon_lidar_admin_4_clipped.geojson", "zone_name_key": "Ward"}
        ],
        "location": {
            "center": [16.7870, 96.1450],
            "initialZoom": 15,
            "bounds": None
        }
    },
    "hcmc_dg": {
        "name": "Thủ Đức District, Hồ Chí Minh City, Vietnam",
        "imagery_metadata": "Digital Globe Imagery",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/HCMC.tif",
        "data_padding": 0,
        "leafletTileLayer": {
            "url": 'tiles/HCMC/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 14,
                "maxNativeZoom": 18,
                "maxZoom": 21,
                "attribution": 'Georeferenced Image'
            }
        },
        "shape_layers": [
            {"name": "Provinces", "shapes_fn": "shapes/hcmc_digital-globe_admin_1_clipped.geojson", "zone_name_key": "NAME_1"},
            {"name": "Districts", "shapes_fn": "shapes/hcmc_digital-globe_admin_2_clipped.geojson", "zone_name_key": "NAME_2"},
            {"name": "Wards", "shapes_fn": "shapes/hcmc_digital-globe_admin_3_clipped.geojson", "zone_name_key": "NAME_3"}
        ],
        "location": {
            "center": [10.838, 106.750],
            "initialZoom": 14,
            "bounds": None
        }
    },
    "airbus": {
        "name": "Virginia, USA",
        "data_layer_type": DatasetTypes.CUSTOM,
        "imagery_metadata": "Airbus Imagery",
        "data_fn": "tiles/airbus_epsg4326.tif",
        "data_padding": 0.003,
        "leafletTileLayer": {
            "url": 'tiles/airbus/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 13,
                "maxNativeZoom": 18,
                "maxZoom": 21,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "Grid", "shapes_fn": "shapes/airbus-data-grid-epsg4326.geojson", "zone_name_key": "id"}
        ],
        "location": {
            "center": [36.80, -76.12],
            "initialZoom": 14,
            "bounds": [[36.882932, -76.2623637], [36.7298842, -76.0249016]]
        }
    },
    "chesapeake": {
        "name": "Maryland, USA",
        "data_layer_type": DatasetTypes.USA_LAYER,
        "imagery_metadata": "NAIP Imagery",
        "data_padding": 20,
        "leafletTileLayer": {
            "url": 'tiles/chesapeake_test/{z}/{x}/{y}.png',
            "args": {
                "tms": True,
                "minZoom": 2,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": None,
        "location": {
            "center": [38.11437, -75.99980],
            "initialZoom": 10,
        }
    },
    "florida_keys_2010_wmts": {
        "name": "Florida Keys, 2010",
        "imagery_metadata": "NAIP 2010",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/florida_keys_2010.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:florida_keys_2010_rgb&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/florida_keys_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [24.7007, -81.3847],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "florida_keys_2013_wmts": {
        "name": "Florida Keys, 2013",
        "imagery_metadata": "NAIP 2013",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/florida_keys_2013.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:florida_keys_2013_rgb&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/florida_keys_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [24.7007, -81.3847],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "florida_keys_2015_wmts": {
        "name": "Florida Keys, 2015",
        "imagery_metadata": "NAIP 2015",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/florida_keys_2015.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:florida_keys_2015_rgb&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/florida_keys_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [24.7007, -81.3847],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_2010_wmts": {
        "name": "LA Coast, 2010",
        "imagery_metadata": "NAIP 2010",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2010.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_2010_rgb&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_2013_wmts": {
        "name": "LA Coast, 2013",
        "imagery_metadata": "NAIP 2013",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2013.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_2013_rgb&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_2015_wmts": {
        "name": "LA Coast, 2015",
        "imagery_metadata": "NAIP 2015",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2015.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_2015_rgb&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_nlcd_wmts": {
        "name": "LA Coast, NLCD Ground Truth Old",
        "imagery_metadata": "Ground Truth",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2013.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_nlcd&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_nlcd_2011_wmts": {
        "name": "LA Coast, NLCD Ground Truth 2011",
        "imagery_metadata": "Ground Truth",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2013.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_nlcd_2011&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_nlcd_2013_wmts": {
        "name": "LA Coast, NLCD Ground Truth 2013",
        "imagery_metadata": "Ground Truth",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2013.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_nlcd_2013&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_nlcd_2016_wmts": {
        "name": "LA Coast, NLCD Ground Truth 2016",
        "imagery_metadata": "Ground Truth",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2013.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_nlcd_2016&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_embed_2010_wmts": {
        "name": "LA Coast, 2010",
        "imagery_metadata": "Embedded NLCD 2010",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2010.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_embed_2010&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    },
    "la_coast_embed_2013_wmts": {
        "name": "LA Coast, 2013",
        "imagery_metadata": "Embedded NLCD 2013",
        "data_layer_type": DatasetTypes.CUSTOM,
        "data_fn": "tiles/la_coast_2013.vrt",
        "data_padding": 20,
        "leafletTileLayer": {
            "wms": False,
            "url": 'http://msrcalebgeoserver.eastus.cloudapp.azure.com:8080/geoserver/gwc/service/wmts?layer=local:la_coast_embed_2013&style=&tilematrixset=EPSG:3857&Service=WMTS&Request=GetTile&Version=1.0.0&Format=image/jpeg&TileMatrix=EPSG:3857:{z}&TileCol={x}&TileRow={y}',
            "args": {
                "tms": False,
                "minZoom": 8,
                "maxNativeZoom": 18,
                "maxZoom": 20,
                "attribution": 'Georeferenced Image',
            }
        },
        "shape_layers": [
            {"name": "NAIP tiles", "shapes_fn": "shapes/la_coast_29090_2010_tiles.geojson", "zone_name_key": "location"}
        ],
        "location": {
            "center": [29.598, -90.559],
            "initialZoom": 10,
            "bounds": None
        }
    }
}


def load_geojson_as_list(fn):
    shapes = []
    areas = []
    crs = None
    with fiona.open(fn) as f:
        src_crs = f.crs
        for row in f:
            geom = row["geometry"]
            if geom["type"] == "Polygon":
                lon, lat = geom["coordinates"][0][0]
            elif geom["type"] == "MultiPolygon":
                lon, lat = geom["coordinates"][0][0][0]
            else:
                raise ValueError("Polygons and MultiPolygons only")

            zone_number = utm.latlon_to_zone_number(lat, lon)
            hemisphere = "+north" if lat > 0 else "+south"
            dest_crs = "+proj=utm +zone=%d %s +datum=WGS84 +units=m +no_defs" % (zone_number, hemisphere)
            projected_geom = fiona.transform.transform_geom(src_crs, dest_crs, geom)
            area = shapely.geometry.shape(projected_geom).area / 1000000.0 # we calculate the area in square meters then convert to square kilometers

            shape = shapely.geometry.shape(geom)
            areas.append(area)
            shapes.append(shape)
    return shapes, areas, src_crs


def get_javascript_string_from_dataset(dataset):
    outputs = {
        "center": dataset["location"]["center"],
        "initialZoom": dataset["location"]["initialZoom"],
        "name": dataset["name"],
        "imageMetadata": dataset["imagery_metadata"],
        "url": dataset["leafletTileLayer"]["url"],
        "kwargs": str(dataset["leafletTileLayer"]["args"]).replace("True","true").replace("False","false"),
        "shapes": str([
            {"name": shape_layer["name"], "shapes_fn": shape_layer["shapes_fn"], "zone_name_key": shape_layer["zone_name_key"]} for shape_layer in dataset["shape_layers"]
        ]).replace("None", "null") if dataset["shape_layers"] is not None else "null"
    }

    if "wms" in dataset["leafletTileLayer"] and dataset["leafletTileLayer"]["wms"]:
        return '''{{
            "location": [{center}, {initialZoom}, "{name}", "{imageMetadata}"],
            "tileObject": L.tileLayer.wms("{url}", {kwargs}),
            "shapes": {shapes}
        }}'''.format(**outputs)
    else:
        return '''{{
            "location": [{center}, {initialZoom}, "{name}", "{imageMetadata}"],
            "tileObject": L.tileLayer("{url}", {kwargs}),
            "shapes": {shapes}
        }}'''.format(**outputs)



'''When this file is loaded we should load each dataset
'''
for dataset_key, dataset in DATASET_DEFINITIONS.items():

    javascript_string = ""

    loaded = True
    shape_layers = {}

    # Load shapes first
    if dataset["shape_layers"] is not None:
        for shape_layer in dataset["shape_layers"]:
            fn = os.path.join(ROOT_DIR, shape_layer["shapes_fn"])
            if os.path.exists(fn):
                shapes, areas, crs = load_geojson_as_list(fn)
                shape_layer["geoms"] = shapes
                shape_layer["areas"] = areas
                shape_layer["crs"] = crs["init"]
                shape_layers[shape_layer["name"]] = shape_layer
            else:
                print("WARNING: %s doesn't exist, this server will not be able to serve the '%s' dataset" % (fn, dataset_key))
                loaded = False


    # Check to see if data_fn exists
    if "data_fn" in dataset:
        fn = os.path.join(ROOT_DIR, dataset["data_fn"])
        if not os.path.exists(fn):
            print("WARNING: %s doesn't exist, this server will not be able to serve the '%s' dataset" % (fn, dataset_key))
            loaded = False


    if loaded:
        if dataset["data_layer_type"] == DatasetTypes.CUSTOM:
            data_loader = DataLoaderCustom(dataset["data_fn"], shape_layers, dataset["data_padding"])
        elif dataset["data_layer_type"] == DatasetTypes.USA_LAYER:
            data_loader = DataLoaderUSALayer(shape_layers, dataset["data_padding"])
        elif dataset["data_layer_type"] == DatasetTypes.BASEMAP:
            data_loader = DataLoaderBasemap(dataset["data_url"], dataset["data_padding"])
        else:
            raise ValueError("DatasetType not recognized")

        DATASETS[dataset_key] = {
            "data_loader": data_loader,
            "shape_layers": shape_layers,
            "javascript_string": get_javascript_string_from_dataset(dataset)
        }
    else:
        pass # we are missing some files needed to load this dataset
