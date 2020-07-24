import os
import io
import threading

import numpy as np

import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logging(log_path, log_name, level=logging.DEBUG):

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    print_formatter = logging.Formatter('[%(asctime)s] - %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')

    logger = logging.getLogger("server")
    logger.setLevel(level)

    file_handler = TimedRotatingFileHandler(log_path + "/%s.txt" % (log_name), when='midnight', interval=1)
    file_handler.suffix = "%Y%m%d"
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    print_handler = logging.StreamHandler()
    print_handler.setFormatter(print_formatter)
    logger.addHandler(print_handler)

    return logger


def serialize(array):
    with io.BytesIO() as f:
        np.save(f, array)
        return f.getvalue()

def deserialize(data):
    with io.BytesIO(data) as f:
        return np.load(f)

class AtomicCounter:
    ''' From https://gist.github.com/benhoyt/8c8a8d62debe8e5aa5340373f9c509c7 '''
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = threading.Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

NLCD_CLASSES = [
    0, 11, 12, 21, 22, 23, 24, 31, 41, 42, 43, 51, 52, 71, 72, 73, 74, 81, 82, 90, 95, 255
]
NLCD_CLASS_TO_IDX = {
    cl: i for i, cl in enumerate(NLCD_CLASSES)
}

COLOR_MAP_NLCD = np.array([
    [0,0,1],
    [1,1,1],
    [0.6,0.6,0.3],
    [0.4,0.4,0.2],
    [0.2,0.2,0.1],
    [0.06,0.06,0.03],
    [0.4,0.4,0.6],
    [0,0.80,0],
    [0,0.55,0],
    [0,0.30,0],
    [0.85,0.85,0.85],
    [0.27,0.60,0.27],
    [0.35,0.76,0.35],
    [0.85,0.85,0.85],
    [0.85,0.85,0.85],
    [0.85,0.85,0.85],
    [0.70,1.00,0.70],
    [0.50,0.70,0.50],
    [0.0,0.55,0.3],
    [0.2,0.90,0.6],
    [1,0,0],
], dtype=np.float32)

COLOR_MAP_LC6 = np.array([
    [0,0,1],
    [0,0.5,0],
    [0.5,1,0.5],
    [0.48,0.48,0.12],
    [0.5,0.375,0.375],
    [0.10,0.10,0.10],
], dtype=np.float32)

COLOR_MAP_LC4 = np.array([
    [0,0,1],
    [0,0.5,0],
    [0.5,1,0.5],
    [0.5,0.375,0.375],
], dtype=np.float32)


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def to_one_hot(im, class_num):
    one_hot = np.zeros((class_num, im.shape[-2], im.shape[-1]), dtype=np.float32)
    for class_id in range(class_num):
        one_hot[class_id, :, :] = (im == class_id).astype(np.float32)
    return one_hot

def to_one_hot_batch(batch, class_num):
    one_hot = np.zeros((batch.shape[0], class_num, batch.shape[-2], batch.shape[-1]), dtype=np.float32)
    for class_id in range(class_num):
        one_hot[:, class_id, :, :] = (batch == class_id).astype(np.float32)
    return one_hot

def class_prediction_to_img(y_pred, hard=True, color_list=None):
    assert len(y_pred.shape) == 3, "Input must have shape (height, width, num_classes)"
    height, width, num_classes = y_pred.shape

    if color_list is None:
        colour_map = COLOR_MAP_LC4
    else:
        new_color_list = []
        for color in color_list:
            color = color.lstrip("#")
            color = tuple(int(color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
            new_color_list.append(color)
        colour_map = np.array(new_color_list)

    if hard:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        y_pred_temp = y_pred.argmax(axis=2)
        for c in range(num_classes):
            mask = (y_pred_temp==c)
            for ch in range(3):
                img[:,:, ch][mask] = int(255*colour_map[c, ch])
    else:
        img = np.zeros((height, width, 3), dtype=np.float32)
        for c in range(num_classes):
            for ch in range(3):
                img[:, :, ch] += y_pred[:, :, c] * colour_map[c, ch]
    return img
    
def nlcd_to_img(img):
    return np.vectorize(NLCD_COLOR_MAP.__getitem__, signature='()->(n)')(img).astype(np.uint8)

def get_shape_layer_by_name(shapes, layer_name):
    for layer in shapes:
        if layer["name"] == layer_name:
            return layer
    return None

def get_random_string(length):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join([alphabet[np.random.randint(0, len(alphabet))] for i in range(length)])