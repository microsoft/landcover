#!/usr/bin/env python
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rasterio

def main():
    '''Usage: python create_raster_preview.py INPUT.tif OUTPUT.jpg [desired width of OUTPUT.jpg in pixels (default: 400)]'''

    if len(sys.argv) < 3:
        print(main.__doc__)
        return

    if len(sys.argv) == 4:
        desired_width = int(sys.argv[3])
    else:
        desired_width = 400

    if len(sys.argv) > 4:
        print(main.__doc__)
        return


    with rasterio.open(sys.argv[1],"r") as f:
        data = np.rollaxis(f.read(), 0, 3)
        assert data.dtype == np.uint8, "We expect the input TIFF to be of the Byte type"
        assert len(data.shape) == 3, "We expect the input TIFF to have 3 dimensions"
        assert data.shape[0] > data.shape[2] and data.shape[1] > data.shape[2], "The input TIFF has a LOT of channels, is this right?"
        assert data.shape[2] >= 3, "We expect the input TIFF to have 3 or more channels (with the first 3 being RGB)"

    height, width, _ = data.shape
    dpi = width / desired_width
    print("Raster width x height: %d x %d" % (width, height))
    print("Using a DPI of %f" % (dpi))
    print("Output width x height: %d x %d" % (width/dpi, height/dpi))

    # The following is how you get matplotlib to make an image without borders
    fig, ax = plt.subplots(1, figsize=(width/dpi, height/dpi), dpi=dpi)
    ax.set_position([0, 0, 1, 1])

    ax.imshow(data[:,:,:3])
    ax.axis("off")

    fig.savefig(
        sys.argv[2],
        bbox_inches=matplotlib.transforms.Bbox([[0, 0], [width/dpi, height/dpi]]),
        dpi=1
    )
    plt.close()

if __name__ == "__main__":
    main()
