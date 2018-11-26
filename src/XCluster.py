import numpy as np
from scipy.spatial import distance


class XCluster:

    def __init__(self, first_vector, first_y):
        self.mean_vector = first_vector
        self.x_vectors = [first_vector]
        self.y_vectors = [first_y]
        self.size = 1
        self.child_nodes = []

    def add_vector(self, new_vect, new_y):
        old_vector = np.array(self.mean_vector) * self.size
        self.mean_vector = (np.array(old_vector) + np.array(new_vect)) / (self.size + 1)
        self.x_vectors.append(new_vect)
        self.y_vectors.append(new_y)
        self.size += 1

    # The paper says to use a n^2 operation...
    def should_split(self, selectivity):
        for y_vector_1 in self.y_vectors:
            for y_vector_2 in self.y_vectors:
                euclid_distance = distance.euclidean(y_vector_1, y_vector_2)
                if euclid_distance > selectivity:
                    return True
        return False

