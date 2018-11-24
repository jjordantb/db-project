import time

import numpy as np
import sys

from numpy.linalg import LinAlgError
from scipy.linalg import orth
from scipy.spatial import distance
from scipy import stats

from XCluster import XCluster
from YCluster import YCluster


def compute_clusters(input_data, num_clusters, sensitivity):
    y_clusters = [YCluster(input_data[0][1])]
    x_clusters = [XCluster(input_data[0][0], input_data[0][1])]
    p = 1
    for i in range(1, len(input_data)):
        # Find the nearest y-cluster to vector i
        y_vector_i = input_data[i][1]
        x_vector_i = input_data[i][0]
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
        if p < num_clusters and dist >= sensitivity:
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
    def build_tree(self, s_prime, selectivity):
        p = self.num_clusters

        # Compute the clusters
        print('Computing Clusters for', self)
        clusters = compute_clusters(s_prime, p, selectivity)
        self.y_clusters = clusters[1]
        self.x_clusters = clusters[0]
        print('Computed', len(self.x_clusters), 'Clusters')

        # Compute the covariance matricies
        print('Computing Matrices for', self)
        for i in range(0, len(self.y_clusters)):
            self.x_clusters[i].compute_cov()

        if len(self.x_clusters) > 1:
            print('Finding Children for', self)
            for i, x_cluster in enumerate(self.x_clusters):
                print('Checking to split', x_cluster, 'in', self)
                start_time = int(round(time.time() * 1000))
                should_split = x_cluster.should_split(1)
                elapsed = int(round(time.time() * 1000)) - start_time
                print('Split Check took', elapsed, 'ms')
                if should_split:
                    new_node = Node(self.num_clusters, self.num_classes, self.depth + 1)
                    x_cluster.child_nodes.append(new_node)
                    cluster_data = x_cluster.get_data()
                    print('Cluster data of', len(cluster_data))
                    new_node.build_tree(cluster_data, selectivity)

    def get_x_centers(self):
        vects = []
        for x_cluster in self.x_clusters:
            vects.append(x_cluster.mean_vector)
        return vects

    def coordinate_vector(self, cluster_index):
        centers = np.array(self.get_x_centers()).transpose()
        m = orth(centers)
        return m.transpose() * self.x_clusters[cluster_index].mean_vector

    def compute_distances_to(self, x):
        dists = []
        for i, x_cluster in enumerate(self.x_clusters):
            dists.append(distance.euclidean(x, x_cluster.mean_vector))
            # dists.append(-stats.multivariate_normal.logpdf(x, x_cluster.mean_vector, x_cluster.cov_matrix, allow_singular=True))
        return dists

    def get_closest_cluster_pair(self, x):
        raw = self.compute_distances_to(x)
        distances = np.array(raw)
        num = min(distances.size, 3 * int(self.depth / 2))
        mins = np.argpartition(distances, -num)[-num:]
        ret = []
        # print(mins, distances)
        for m in mins:
            index = raw.index(distances[m])
            # print('idx', index)
            ret.append([distances[m], self.x_clusters[index], self.y_clusters[index]])
        # print('Return', ret)
        return ret
