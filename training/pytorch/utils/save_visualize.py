import numpy as np
import torch
import einops


RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
BLACK = [0, 0, 0]

WATER = BLUE
FIELD = GREEN
TREES = [0, 200, 0]
BUILT = [128, 128, 128]
UNKNOWN = BLACK


def save_visualize(inputs, outputs, ground_truth):
    # inputs: (batch_size, channels_in, height_outer, width_outer)
    # outputs: (batch_size, channels_out, height_inner, )
    # ground_truth: 
    
    batch_size, channels_output, height_output, width_output = outputs.shape
    batch_size, channels_input,  height_input,  width_input  = inputs.shape
    batch_size,                  height_output, width_output = ground_truth.shape

    outputs = outputs.argmax(dim=1)
    # (batch_size, height_out, width_out)

    margin = (width_input - width_output) / 2
    
    channels = 3
    rgb_outputs = np.zeros((batch_size, channels, height, width))

    
    
    pdb.set_trace()
    # Dump visualization of predictions
    # save to hyper_parameters['predictions_path']
    for (output_num, output) in outputs:
        np.save(outputs, str(hyper_parameters['predictions_path']) + '_output_[%d]_epoch_[%d].npy' % (output_num, epoch))


def crop_to_smallest_dimensions(small_tensor, large_tensor):
    '''
    Return a `small_tensor`-sized slice of `large_tensor`, such that the slice has an equal margin on all sides.

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
>>> crop_to_smallest_dimensions(x, y)
array([[2., 2., 2., 2.],
       [2., 2., 8., 2.],
       [2., 2., 2., 2.],
       [2., 2., 2., 9.]])
    '''
    if len(small_tensor.shape) != len(large_tensor.shape):
        raise Exception('small_tensor and large_tensor must have same number of dimensions')
    
    margins = [0] * len(small_tensor.shape)
    for d in range(len(small_tensor.shape)):
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

    new_tensor = large_tensor[np.ix_(*ranges)]
    
    return new_tensor

        

def center_in_larger_dimensions(small_tensor, large_tensor):
    '''
    Pad `small_tensor` with zeros to meet the dimensions of `large_tensor`, keeping an equal padding on all sides

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
    
    new_shape = list(large_tensor.shape)
    margins = [0] * len(new_shape)
    for d in range(len(new_shape)):
        margins[d] = (large_tensor.shape[d] - small_tensor.shape[d]) / 2
        if margins[d].is_integer():
            margins[d] = int(margins[d])
        else:
            raise Exception('Cannot symmetrically embed tensor of shape %s inside of %s in dimensions %s:'
                            'dimension %d gives a margin that is not a whole number (%d - %d) / 2 == %f' %
                            (small_tensor.shape, large_tensor.shape, dimension_indices, d, small_tensor.shape[d], large_tensor.shape[d], (large_tensor.shape[d] - small_tensor.shape[d]) / 2 ))
    new_tensor = np.zeros(new_shape)
    
    ranges = [range(margins[d],
                    large_tensor.shape[d] - margins[d])
              for d in range(len(small_tensor.shape))]

    new_tensor[np.ix_(*ranges)] = small_tensor
    
    return new_tensor

