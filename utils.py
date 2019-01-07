import numpy as np
import string


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

def id_generator(size=10, chars=list(string.ascii_uppercase + string.digits)):
    return ''.join([np.random.choice(chars) for _ in range(size)])

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

def class_prediction_to_img(y_pred, hard=True):
    assert len(y_pred.shape) == 3, "Input must have shape (height, width, num_classes)"
    height, width, num_classes = y_pred.shape

    if num_classes > 10:
        colour_map = COLOR_MAP_NLCD
    elif num_classes == 6:
        colour_map = COLOR_MAP_LC6
    elif num_classes == 4:
        colour_map = COLOR_MAP_LC4
    else:
        raise TypeError("num_classes must match a known prediction format")

    img = np.zeros((height, width, 3), dtype=np.float32)

    y_pred_temp = y_pred.copy()
    if hard:
        y_pred_temp = y_pred.argmax(axis=2)
        for c in range(num_classes):
            for ch in range(3):
                img[:,:, ch] += (y_pred_temp == c) * colour_map[c, ch]
    else:
        for c in range(num_classes):
            for ch in range(3):
                img[:, :, ch] += y_pred_temp[:, :, c] * colour_map[c, ch]
    return img
    
#def nlcd_to_img(img):
#    return np.vectorize(NLCD_COLOR_MAP.__getitem__, signature='()->(n)')(img).astype(np.uint8)
