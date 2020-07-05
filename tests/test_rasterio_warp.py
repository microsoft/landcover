import sys
sys.path.append("../")

import rasterio
import rasterio.warp
import numpy as np

from web_tool.DataLoader import DataLoaderCustom, warp_data_to_3857

dataloader = DataLoaderCustom("../data/imagery/hcmc_sentinel.tif", [], padding=1100)

extent = {
    "spatialReference": {"latestWkid": 3857},
    "xmax": 11880604,
    "xmin": 11880304,
    "ymax": 1206861.0000000007,
    "ymin": 1206561
}

patch, crs, transform, bounds = dataloader.get_data_from_extent(extent)
patch = np.rollaxis(patch, 0, 3)

print(patch.shape)

patch, new_bounds = warp_data_to_3857(patch, crs, transform, bounds)