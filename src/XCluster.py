import numpy as np


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
