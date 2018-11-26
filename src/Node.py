import sys
import time

import numpy as np
import math
from scipy.spatial import distance

from XCluster import XCluster
from YCluster import YCluster


def compute_clusters(x_train, y_train, num_clusters, selectivity):
    y_clusters = [YCluster(y_train[0])]
    x_clusters = [XCluster(x_train[0], y_train[0])]
    p = 1
    for i in range(1, len(x_train)):
        # Find the nearest y-cluster to vector i
        y_vector_i = y_train[i]
        x_vector_i = x_train[i]
        dist = sys.maxsize
        closest_cluster = None
        index = 0
        for i, cluster in enumerate(x_clusters):
            tmp_dist = distance.euclidean(np.array(cluster.mean_vector), np.array(x_vector_i))
            if tmp_dist < dist:
                dist = tmp_dist
                closest_cluster = cluster
                index = i
            i += 1
        if p < num_clusters and dist >= selectivity:
            new_y_cluster = YCluster(y_vector_i)
            y_clusters.append(new_y_cluster)
            new_x_cluster = XCluster(x_vector_i, y_vector_i)
            x_clusters.append(new_x_cluster)
            p += 1
        else:
            closest_cluster.add_vector(x_vector_i, y_vector_i)
            y_clusters[index].add_vector(y_vector_i)
    return [x_clusters, y_clusters]


class Node:

    def __init__(self, num_clusters, num_classes, depth):
        self.x_clusters = None  # init set of x_clusters
        self.y_clusters = None  # init set of y_clusters
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.depth = depth
        print('At Tree Depth', self.depth)

    # s_prime -> set of tuples (x, y)
    def build_tree(self, x_train, y_train, selectivity):
        # Compute the clusters
        print('Computing Clusters for', self)
        clusters = compute_clusters(x_train, y_train, self.num_clusters, selectivity)
        self.y_clusters = clusters[1]
        self.x_clusters = clusters[0]
        print('Computed', len(self.x_clusters), 'Clusters')

        if len(self.x_clusters) > 1:
            print('Finding Children for', self)
            for i, x_cluster in enumerate(self.x_clusters):
                print('Checking to split', x_cluster, 'in', self)
                start_time = int(round(time.time() * 1000))
                should_split = x_cluster.should_split(0)
                elapsed = int(round(time.time() * 1000)) - start_time
                print('Split Check took', elapsed, 'ms')
                if should_split:
                    new_node = Node(self.num_clusters, self.num_classes, self.depth + 1)
                    x_cluster.child_nodes.append(new_node)
                    print('Cluster data of', len(x_cluster.x_vectors))
                    new_node.build_tree(x_cluster.x_vectors, x_cluster.y_vectors, selectivity)

    def compute_distances_to(self, x):
        dists = []
        for i, x_cluster in enumerate(self.x_clusters):
            dists.append(-distance.euclidean(x, x_cluster.mean_vector))
        return dists

    def get_closest_cluster_pairs(self, x, k):
        raw = self.compute_distances_to(x)
        distances = np.array(raw)
        # num = distances.size
        num = min(distances.size, int(math.pow(k, self.depth)))
        mins = np.argpartition(distances, -num)[-num:]
        ret = []
        for m in mins:
            index = raw.index(distances[m])
            ret.append([distances[m], self.x_clusters[index], self.y_clusters[index]])
        return ret
