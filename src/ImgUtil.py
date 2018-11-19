import os

from array import array


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
