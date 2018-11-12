import os

import numpy as np
from array import array


# Converts a 1D array of min-max values to a covariance matrix
# img_dim - dimensions of the image, scale to square
# min-max are bounding values of the pixel
def image_to_cov(vector, img_dim):
    arr = np.reshape(np.array(vector), (-1, img_dim))  # convert to numpy array and reshape it to a 2d array
    return np.cov(arr, rowvar=False, bias=True)  # convert to covariance matrix


# reads a mnist file
# returns an list of vectors
def load_mnist(num):
    file_bytes_as_ints = array("B")
    with (open('../data/mnist/data' + str(num), "rb")) as f:
        file_bytes_as_ints.fromfile(f, os.path.getsize(f.name))
    imgs = []
    for i in range(0, int(len(file_bytes_as_ints) / 784)):
        tmp = []
        for j in range(0, 784):
            tmp.append(file_bytes_as_ints[i + j])
        imgs.append(tmp)
    return imgs
