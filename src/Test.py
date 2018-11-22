# import ImgUtil
# import numpy as np
# import scipy.misc as smp
#
# from YCluster import YCluster
#
# imgs0 = ImgUtil.load_mnist(0)
# cluster = YCluster(imgs0[0])
# cluster.add_vector(imgs0[1])
# cov = np.cov([imgs0[0], imgs0[1]], rowvar=False)
# print(cov)
# print(cluster.cov_matrix)

# print(vals)
# print(cov)
# cov2 = np.cov([np.cov([imgs0[0], imgs0[1]], rowvar=False), imgs0[2]], rowvar=False)
# print(cov2)
import random
import sys
import ImgUtil
from Node import Node
from scipy import stats
import time

data = []  # tuple of (image data vector, 1-hot-vector)
for i in range(0, 10):
    imgs = ImgUtil.load_mnist(i)
    vect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vect[i] = 1
    for img in imgs:
        data.append((img, vect))


def test(n, x_in):
    closest_cluster_pair = n.get_closest_cluster_pair(x_in)
    dist = closest_cluster_pair[0]
    if len(closest_cluster_pair[1].child_nodes) > 0:
        for child in closest_cluster_pair[1].child_nodes:
            return test(child, x_in)
    return closest_cluster_pair
    # distance = sys.maxsize
    # index = 0
    # i = 0
    # for x_cluster in n.x_clusters:
    #     if len(x_cluster.child_nodes) > 0:
    #         for child in x_cluster.child_nodes:
    #             child_result = test(child, x_in)
    #             if child_result[0] < distance:
    #                 distance = child_result[0]
    #     else:
    #         tmp_distance = -stats.multivariate_normal.logpdf(x_in, x_cluster.mean_vector, x_cluster.cov_matrix)
    #         if tmp_distance < distance:
    #             distance = tmp_distance
    #             index = i
    #     i += 1
    # return [distance, node.y_clusters[index].mean_vector]


node = Node(10, 10)
node.build_tree(data, 1)
# print(node.coordinate_vector(0).shape)
# print(node.linear_manifold(data[0][0]).shape)
for i in range(0, 10):
    actual = i
    val = ImgUtil.load_mnist(actual)
    v = val[random.randint(0, 50)]
    start_time = int(round(time.time() * 1000))
    pred = test(node, v)[2].mean_vector
    elapsed = int(round(time.time() * 1000)) - start_time
    print('Actual ' + str(actual) + ' Prediction ' + str(pred), 'in', elapsed, 'ms')

# print(v)

