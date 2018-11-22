import numpy as np
from scipy.linalg import orth


class XCluster:

    def __init__(self, first_vector):
        self.mean_vector = first_vector
        self.cov_matrix = None
        self.vectors = [first_vector]
        self.size = 1

    def add_vector(self, new_vect):
        old_vector = np.array(self.mean_vector) * self.size
        self.mean_vector = (np.array(old_vector) + np.array(new_vect)) / (self.size + 1)
        self.vectors.append(new_vect)
        self.size += 1

    def compute_cov(self):
        self.cov_matrix = np.cov(self.vectors, rowvar=False)

    def snll(self, x):
        v1 = np.array(x) - self.mean_vector
        return 0.5 * v1.transpose() * np.linalg.inv(self.cov_matrix) * v1 + 0.5 * np.log(self.cov_matrix.size)

    def linear_manifold(self, x):
        vectors = np.array(self.vectors).transpose()
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
