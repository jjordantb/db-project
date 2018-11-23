import numpy as np
from scipy.linalg import orth
from scipy.spatial import distance


class XCluster:

    def __init__(self, first_vector, first_y):
        self.mean_vector = first_vector
        self.cov_matrix = None
        self.vectors = [first_vector]
        self.y_vectors = [first_y]
        self.size = 1
        self.child_nodes = []

    def add_vector(self, new_vect, new_y):
        old_vector = np.array(self.mean_vector) * self.size
        self.mean_vector = (np.array(old_vector) + np.array(new_vect)) / (self.size + 1)
        self.vectors.append(new_vect)
        self.y_vectors.append(new_y)
        self.size += 1

    def compute_cov(self):
        self.cov_matrix = np.cov(self.vectors, rowvar=False)

    # The paper says to use a n^2 operation...
    def should_split(self, selectivity):
        for y_vector_1 in self.y_vectors:
            for y_vector_2 in self.y_vectors:
                euclid_distance = distance.euclidean(y_vector_1, y_vector_2)
                print(euclid_distance)
                if euclid_distance > selectivity:
                    return True
        return False

    def get_data(self):
        data = []
        for i, vector in enumerate(self.vectors):
            data.append([vector, self.y_vectors[i]])
        return data

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
