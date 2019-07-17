import collections

import matplotlib
matplotlib.use("Agg") 
import matplotlib.cm

import numpy as numpy

import cv2
import mercantile

class Heatmap():
    count_dict = collections.defaultdict(int)
    cmap = matplotlib.cm.get_cmap("Reds")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=20, clip=True)

    @staticmethod
    def increment(z,y,x):
        #print("Incrementing", (x,y,z))
        while z > 1:
            key = (z,y,x)
            Heatmap.count_dict[key] += 1
            tile = mercantile.Tile(x,y,z)
            tile = mercantile.parent(tile)
            x,y,z = tile.x, tile.y, tile.z

    @staticmethod
    def get(z,y,x):
        key = (z,y,x)
        val = Heatmap.count_dict[key]
        img = np.zeros((256,256,4), dtype=np.uint8)
        if val != 0:
            img[:,:] = np.round(np.array(Heatmap.cmap(Heatmap.norm(val))) * 255).astype(int)

        img = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))[1].tostring()
        return img

    @staticmethod
    def reset():
        Heatmap.count_dict = collections.defaultdict(int)