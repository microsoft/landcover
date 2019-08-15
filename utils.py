import numpy as np


def one_hot_from_indices(index_ids):
    return np.eye(index_ids.max() + 1)[index_ids]


def make_one_hot(array, axis=-1):
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
