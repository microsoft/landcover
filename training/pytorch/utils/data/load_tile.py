import numpy as np

def load_tile(tile_file_name):
    tile = np.load(tile_file_name)
    tile = collapse_classes(tile)

    return tile

def collapse_classes(tile):
    # tile: (batch, channels, row, col)
    # channels:
    #  0 - 3:  R, G, B, IR
    #  4:      assigned class label
    
    # Focus on the "class" channel
    y_train_hr = tile[0, 4, :, :]

    # Collapse classes to 0 - 4
    y_train_hr[y_train_hr == 15] = 0
    y_train_hr[y_train_hr == 5] = 4
    y_train_hr[y_train_hr == 6] = 4
    
    return tile

def features(tile):
    return tile[:, :4, :, :]

def labels(tile):
    return tile[:, 4, :, :]
