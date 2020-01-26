import numpy as np


def one_hot_from_indices(index_ids):
    '''
    Turn an n-dimensional tensor of integers into an (n+1)-dimensional array of one-hot vectors, containing a 1 in the given index and 0 everywhere else.
    '''
    return np.eye(index_ids.max() + 1)[index_ids]


def soft_to_hard(array, axis=-1):
    '''
    Turn an array of floats (perhaps probabilities) into a one-hot vector, with each vector along `axis` having its max value turned into a 1, and the rest turned to 0s.

    Only tested for axis=-1, as this is usually what we want to do.
    '''
    return one_hot_from_indices(array.argmax(axis=axis))


def test_one_hot_from_indices():
    arr = np.array([[0, 1, 3], [2, 1, 0]])
    assert (one_hot_from_indices(arr) == np.array(
        [[[1., 0., 0., 0.],
          [0., 1., 0., 0.],
          [0., 0., 0., 1.]],
         
         [[0., 0., 1., 0.],
          [0., 1., 0., 0.],
          [1., 0., 0., 0.]]])).all()


def test_make_one_hot():
    assert (make_one_hot(np.array(
        [[[0.9,  0.,  0.1,  0.],
          [0.1,  0.8, 0.1,  0.],
          [0.2,  0.1, 0.1,  0.6]],
         
         [[0.,    0.,   0.8,   0.1],
          [-0.2,  0.,  -0.1,  -0.1],
          [1.5,   0.,   0.,    0.]]])) == \
            
            np.array(
        [[[1., 0., 0., 0.],
          [0., 1., 0., 0.],
          [0., 0., 0., 1.]],
         
         [[0., 0., 1., 0.],
          [0., 1., 0., 0.],
          [1., 0., 0., 0.]]])).all()
