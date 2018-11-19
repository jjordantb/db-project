import numpy as np
import sys

from XCluster import XCluster
from YCluster import YCluster


def computer_y_clusters(y_vectors, num_clusters, sensitivity):
    y_clusters = [YCluster(y_vectors[0])]
    p = 1
    for i in range(1, len(y_vectors)):
        # Find the nearest y-cluster to vector i
        vector_i = y_vectors[i]
        dist = sys.maxsize
        closest_cluster = None
        for cluster in y_clusters:
            tmp_dist = np.linalg.norm(np.array(cluster.mean_vector) - np.array(vector_i))
            if tmp_dist < dist:
                dist = tmp_dist
                closest_cluster = cluster
        if p < num_clusters and dist >= sensitivity:
            new_cluster = YCluster(vector_i)
            y_clusters.append(new_cluster)
            p += 1
        else:
            closest_cluster.add_vector(vector_i)
    return y_clusters


class Node:

    def __init__(self, num_clusters):
        self.x_clusters = None  # init set of x_clusters
        self.y_clusters = None  # init set of y_clusters
        self.num_clusters = num_clusters  # number of classes / labels

    # s_prime -> set of tuples (x, y)
    def build_tree(self, s_prime, selectivity):
        p = self.num_clusters
        self.y_clusters = computer_y_clusters([s[1] for s in s_prime], p, selectivity)
        self.x_clusters = []
        index = 0
        for y_cluster in self.y_clusters:
            for tuple in s_prime:
                if y_cluster.contains_vector(tuple[1]):
                    if len(self.x_clusters) <= index:
                        self.x_clusters.append(XCluster(tuple[0]))
                    else:
                        self.x_clusters[index].add_vector(tuple[0])
            index += 1
            print(index)
