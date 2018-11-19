import ImgUtil
import numpy as np
import scipy.misc as smp

from YCluster import YCluster

imgs0 = ImgUtil.load_mnist(0)
cluster = YCluster(imgs0[0])
cluster.add_vector(imgs0[1])
cov = np.cov([imgs0[0], imgs0[1]], rowvar=False)
print(cov)
print(cluster.cov_matrix)

# print(vals)
# print(cov)
# cov2 = np.cov([np.cov([imgs0[0], imgs0[1]], rowvar=False), imgs0[2]], rowvar=False)
# print(cov2)
# import sys
#
# from Node import Node
# import ImgUtil
# import numpy as np
#
# data = []  # tuple of (image data vector, 1-hot-vector)
# for i in range(0, 10):
#     imgs = ImgUtil.load_mnist(i)
#     vect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     vect[i] = 1
#     for img in imgs:
#         data.append((img, vect))
#
#
# def test(node, x_in):
#     closest = None
#     distance = sys.maxsize
#     index = 0
#     i = 0
#     for x_cluster in node.x_clusters:
#         tmp_distance = np.linalg.norm(x_cluster.cov_matrix - np.cov(x_in, rowvar=False))
#         if tmp_distance < distance:
#             closest = x_cluster
#             distance = tmp_distance
#             index = i
#         i += 1
#     print(index)
#     return node.y_clusters[index].mean_vector
#
#
# node = Node(10)
# node.build_tree(data, 1)
# print(len(node.x_clusters))
# print(test(node, data[0][0]))

