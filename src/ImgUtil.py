import os
import random

from array import array
import numpy as np
import scipy.misc as smp


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
    random.shuffle(imgs)
    return imgs


def draw_image(pixel_array, width, height):
    pixel_array = np.array(pixel_array).reshape(width, height)
    img = smp.toimage(pixel_array, mode='L')
    img.show()
