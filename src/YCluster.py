import numpy as np


class YCluster:

    def __init__(self, first_vector):
        self.mean_vector = first_vector
        self.cov_matrix = first_vector
        self.vectors = [first_vector]
        self.size = 1

    def add_vector(self, new_vect):
        old_vector = np.array(self.mean_vector) * self.size
        self.mean_vector = (np.array(old_vector) + np.array(new_vect)) / (self.size + 1)
        # variance = np.cov(new_vect, rowvar=False)
        # self.cov_matrix = (variance + self.mean_vector) / (self.size + 1)
        self.vectors.append(new_vect)
        self.cov_matrix = np.cov(self.vectors, rowvar=False)
        self.size += 1

    def contains_vector(self, vector):
        return vector in self.vectors
