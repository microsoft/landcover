import os
import numpy as np
import torch
import einops
from PIL import Image
from pathlib import Path

import pdb


RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]

CLASS_TO_COLOR = {
    0:  BLACK,             # UNKNOWN
    1:  BLUE,              # WATER
    2:  [0, 200, 0],       # TREES
    3:  GREEN,             # FIELD
    4:  [128, 128, 128]    # BUILT
}


def save_visualize(inputs, outputs, ground_truth, path):
    batch_size, channels_output, height_output, width_output = outputs.shape
    batch_size, channels_input,  height_input,  width_input  = inputs.shape
    batch_size,                  height_output, width_output = ground_truth.shape

    output_classes = outputs.argmax(dim=1)
    # (batch_size, height_output, width_output)
    # ground_truth has same shape as outputs

    sanitized_inputs = inputs_to_rgb(inputs)
    
    cropped_inputs = crop_to_smallest_dimensions(sanitized_inputs, outputs, (2, 3))
    outputs_color = classes_to_rgb(output_classes, CLASS_TO_COLOR)
    ground_truth_color = classes_to_rgb(ground_truth, CLASS_TO_COLOR)

    # save cropped_inputs, outputs, ground_truth
    save_batch(sanitized_inputs, path, 'input')
    save_batch(cropped_inputs, path, 'cropped_input')
    save_batch(outputs_color, path, 'predictions')
    save_batch(ground_truth_color, path, 'ground_truth')
    

def save_batch(batch, path, file_name_prefix):
    # batch: (batch_size, RGB_channels, height, width)
    for i, image_tensor in enumerate(batch):
        image_numpy = image_tensor.cpu().numpy()
        image_numpy = einops.rearrange(image_numpy, 'rgb height width -> height width rgb')
        os.makedirs(path, exist_ok=True)
        Image.fromarray(image_numpy).save(str(Path(path) / ('%s_%d.png' % (file_name_prefix, i))))

        
def inputs_to_rgb(inputs):
    # inputs: (batch_size, channels, height, width)
    # Convert channels from [R, G, B, IR] --> [R, G, B]
    # Convert pixel values from [0.0, 1.0] --> [0, 255]
    return torch.as_tensor(
        inputs[:, :3, :, :] * 255,
        dtype=torch.uint8)


def classes_to_rgb(predictions, color_map):
    # predictions: (batch_size, height, width)
    # return: (batch_size, rgb_channels, height_width)
    predictions = torch.as_tensor(predictions, dtype=torch.uint8)
    batch_size, height, width = predictions.shape
    
    rgb_channels = 3
    rgb_outputs = torch.zeros((batch_size, height, width, rgb_channels), dtype=torch.uint8)

    for class_type in color_map:
        color = torch.tensor(color_map[class_type], dtype=torch.uint8)
        rgb_outputs[predictions == class_type] = color
    
    rgb_outputs = einops.rearrange(rgb_outputs, 'batch height width rgb -> batch rgb height width')
    return rgb_outputs


def test_convert_to_rgb():
    CLASS_TO_COLOR = {
        0:  BLACK,             # UNKNOWN
        1:  BLUE,              # WATER
        2:  [0, 200, 0],       # TREES
        3:  GREEN,             # FIELD
        4:  [128, 128, 128]    # BUILT
    }

    x = np.zeros((1, 3, 3))
    x[0,0,0] = 1
    x[0,1,1] = 2
    y = to_rgb(x, CLASS_TO_COLOR)
    print(y)
    assert y == np.array(
        [[[[  0,   0,   0,]
           [  0,   0,   0,]
           [  0,   0,   0,]]

          [[  0,   0,   0,]
           [  0, 200,   0,]
           [  0,   0,   0,]]

          [[255,   0,   0,]
           [  0,   0,   0,]
           [  0,   0,   0,]]]])
    
    
        
def crop_to_smallest_dimensions(large_tensor, small_tensor, dimension_indices):
    '''
    Return a `small_tensor`-sized slice of `large_tensor`, such that the slice has an equal margin on all sides.

>>> import numpy as np
>>> from training.pytorch.utils.save_visualize import *
>>> x = np.ones((4, 4))
>>> y = np.ones((8, 8)) * 2
>>> y[5, 5] = 9
>>> y[3, 4] = 8
>>> x
array([[1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.],
       [1., 1., 1., 1.]])
>>> y
array([[2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 8., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 9., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2.]])
>>> crop_to_smallest_dimensions(y, x, (0, 1))
array([[2., 2., 2., 2.],
       [2., 2., 8., 2.],
       [2., 2., 2., 2.],
       [2., 2., 2., 9.]])
    '''
    if len(small_tensor.shape) != len(large_tensor.shape):
        raise Exception('small_tensor and large_tensor must have same number of dimensions')
    
    margins = [0] * len(large_tensor.shape)
    for d in dimension_indices:
        margins[d] = (large_tensor.shape[d] - small_tensor.shape[d]) / 2
        if margins[d].is_integer():
            margins[d] = int(margins[d])
        else:
            raise Exception('Cannot symmetrically embed tensor of shape %s inside of %s in dimensions %s:'
                            'dimension %d gives a margin that is not a whole number (%d - %d) / 2 == %f' %
                            (small_tensor.shape, large_tensor.shape, dimension_indices, d, small_tensor.shape[d], large_tensor.shape[d], (large_tensor.shape[d] - small_tensor.shape[d]) / 2 ))

    ranges = [range(margins[d],
                    large_tensor.shape[d] - margins[d])
              for d in range(len(small_tensor.shape))]

    cropped_tensor = large_tensor[np.ix_(*ranges)]
    
    return cropped_tensor

        

def center_in_larger_dimensions(small_tensor, large_tensor, dimension_indices):
    '''
    Pad `small_tensor` with zeros to meet the dimensions of `large_tensor`, keeping an equal padding on all sides

>>> import numpy as np
>>> from training.pytorch.utils.save_visualize import *
>>> x = np.ones((5, 5))
>>> y = np.ones((9, 9)) * 2
>>> x
array([[1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.],
       [1., 1., 1., 1., 1.]])
>>> y
array([[2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2.],
       [2., 2., 2., 2., 2., 2., 2., 2., 2.]])

>>> center_in_larger_dimensions(x, y, (0, 1))
array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 0., 1., 1., 1., 1., 1., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0.]])
    '''
    if len(small_tensor.shape) != len(large_tensor.shape):
        raise Exception('small_tensor and large_tensor must have same number of dimensions')
    
    new_shape = list(small_tensor.shape)
    margins = [0] * len(new_shape)
    for d in dimension_indices:
        new_shape[d] = large_tensor.shape[d]
        margins[d] = (large_tensor.shape[d] - small_tensor.shape[d]) / 2
        if margins[d].is_integer():
            margins[d] = int(margins[d])
        else:
            raise Exception('Cannot symmetrically embed tensor of shape %s inside of %s in dimensions %s:'
                            'dimension %d gives a margin that is not a whole number (%d - %d) / 2 == %f' %
                            (small_tensor.shape, large_tensor.shape, dimension_indices, d, small_tensor.shape[d], large_tensor.shape[d], (large_tensor.shape[d] - small_tensor.shape[d]) / 2 ))

    ranges = [range(margins[d],
                    large_tensor.shape[d] - margins[d])
              for d in range(len(small_tensor.shape))]

    enlarged_tensor = np.zeros(new_shape)
    enlarged_tensor[np.ix_(*ranges)] = small_tensor
    
    return enlarged_tensor

