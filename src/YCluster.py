import numpy as np


class YCluster:

    def __init__(self, first_vector):
        self.mean_vector = first_vector
        self.cov_matrix = None
        self.size = 1

    def add_vector(self, new_vect):
        old_vector = np.array(self.mean_vector) * self.size
        self.mean_vector = (np.array(old_vector) + np.array(new_vect)) / (self.size + 1)
        self.size += 1
