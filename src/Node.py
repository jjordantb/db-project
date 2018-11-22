import numpy as np
import sys
from scipy.linalg import orth
from scipy.spatial import distance

from XCluster import XCluster
from YCluster import YCluster


def compute_clusters(input_data, num_clusters, sensitivity):
    y_clusters = [YCluster(input_data[0][1])]
    x_clusters = [XCluster(input_data[0][0])]
    p = 1
    for i in range(1, len(input_data)):
        # Find the nearest y-cluster to vector i
        y_vector_i = input_data[i][1]
        x_vector_i = input_data[i][0]
        dist = sys.maxsize
        closest_cluster = None
        index = 0
        i = 0
        for cluster in y_clusters:
            tmp_dist = distance.euclidean(np.array(cluster.mean_vector), np.array(y_vector_i))
            if tmp_dist < dist:
                dist = tmp_dist
                closest_cluster = cluster
                index = i
            i += 1
        if p < num_clusters and dist >= sensitivity:
            new_y_cluster = YCluster(y_vector_i)
            y_clusters.append(new_y_cluster)
            new_x_cluster = XCluster(x_vector_i)
            x_clusters.append(new_x_cluster)
            p += 1
        else:
            closest_cluster.add_vector(y_vector_i)
            x_clusters[index].add_vector(x_vector_i)
    return [x_clusters, y_clusters]


class Node:

    def __init__(self, num_clusters):
        self.x_clusters = None  # init set of x_clusters
        self.y_clusters = None  # init set of y_clusters
        self.num_clusters = num_clusters  # number of classes / labels

    # s_prime -> set of tuples (x, y)
    def build_tree(self, s_prime, selectivity):
        p = self.num_clusters

        # Compute the clusters
        clusters = compute_clusters(s_prime, p, selectivity)
        self.y_clusters = clusters[1]
        self.x_clusters = clusters[0]

        # Compute the covariance matricies
        for i in range(0, len(self.y_clusters)):
            self.y_clusters[i].compute_cov()
            self.x_clusters[i].compute_cov()

    def get_x_centers(self):
        vects = []
        for x_cluster in self.x_clusters:
            vects.append(x_cluster.mean_vector)
        return vects

    def linear_manifold(self, x):
        vectors = np.array(self.get_x_centers()).transpose()
        center = np.mean(vectors, axis=1)
        scatter_vectors = []
        for v in vectors.T:
            scatter_vectors.append(np.array(v) - np.array(center))
        manifold = np.array(vectors) + np.array(scatter_vectors).transpose()
        orthog = orth(manifold)
        scatter_part = center - np.array(x)
        feature_vector = []
        for a in orthog.T:
            feature_vector.append(scatter_part.transpose() * a)
        return np.array(feature_vector).transpose()

    def coordinate_vector(self, cluster_index):
        centers = np.array(self.get_x_centers()).transpose()
        m = orth(centers)
        return m.transpose() * self.x_clusters[cluster_index].mean_vector

