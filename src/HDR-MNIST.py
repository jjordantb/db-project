import random

import ImgUtil
from ImageStore import ImageStore
from Node import Node
import time
import sys
import numpy as np
from scipy.spatial import distance

# Parameters Start
training_test_split = 0.95
nodes = 50
selectivity = 850
k = 3
# Parameters End


class MnistStore(ImageStore):

    def __init__(self, idx, split):
        super().__init__(None)
        self.images = ImgUtil.load_mnist(idx)
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        vect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        vect[idx] = 1
        size = len(self.images)
        for i, img in enumerate(self.images):
            if i >= size * split:
                self.x_test.append(img)
                self.y_test.append(vect)
            else:
                self.x_train.append(img)
                self.y_train.append(vect)

    def fetch_random_image(self):
        rand_image = self.images[random.randrange(0, len(self.images))]
        return rand_image

    def fetch_random_raw_image(self):
        return self.images[random.randrange(0, len(self.images))]


image_stores = []
training = []  # tuple of (image data vector, 1-hot-vector)
testing = []
# Init our training data
for i in range(0, 10):
    store = MnistStore(i, training_test_split)
    for j, x in enumerate(store.x_train):
        training.append((x, store.y_train[j]))
    for j, x in enumerate(store.x_test):
        testing.append((x, store.y_test[j]))
    image_stores.append(store)


def current_time_ms():
    return int(round(time.time() * 1000))


def traverse_tree(n, x_in, terminals):
    closest_cluster_pair = n.get_closest_cluster_pair(x_in, k)
    if len(closest_cluster_pair) > 0:
        for c in closest_cluster_pair:
            if len(c[1].child_nodes) > 0:
                for child in c[1].child_nodes:
                    traverse_tree(child, x_in, terminals)
            else:
                terminals.append(c)


start_training = current_time_ms()
print('Training on', len(training), 'samples')
node = Node(nodes, 10, 1)
node.build_tree(training, selectivity)
end_training = current_time_ms()
print('Training took', end_training - start_training, 'ms')

if 'test' not in sys.argv:
    while True:
        print('What MNIST image would you like to make inference on: ')
        idx = input()
        print('Grabbing random image ' + idx + ' from ImageStore')
        test = image_stores[int(idx)].fetch_random_raw_image()
        start_time = int(round(time.time() * 1000))
        results = []
        traverse_tree(node, test[0], results)
        closest = None
        for result in results:
            dist = distance.euclidean(np.array(result[1].mean_vector), test[0])
            # print('DISTANCE', dist, '->', result[2].mean_vector)
            if closest is None or closest[0] > dist:
                result[0] = dist
                closest = result

        elapsed = int(round(time.time() * 1000)) - start_time
        pred = closest[2].mean_vector
        pred_i = pred.index(max(pred)) if isinstance(pred, list) else pred.tolist().index(max(pred))
        print('The predicted label was', pred_i, 'Here is a random image that is also a', pred_i)
        ImgUtil.draw_image(image_stores[int(idx)].fetch_random_raw_image(), 28, 28)
else:
    print('Testing', len(testing), 'data points')
    correct = 0
    done = 0
    total_time = 0
    for test in testing:
        start_time = int(round(time.time() * 1000))
        results = []
        traverse_tree(node, test[0], results)
        closest = None
        for result in results:
            dist = distance.euclidean(np.array(result[1].mean_vector), test[0])
            # print('DISTANCE', dist, '->', result[2].mean_vector)
            if closest is None or closest[0] > dist:
                result[0] = dist
                closest = result

        elapsed = int(round(time.time() * 1000)) - start_time
        pred = closest[2].mean_vector
        test_i = test[1].index(1)
        pred_i = pred.index(max(pred)) if isinstance(pred, list) else pred.tolist().index(max(pred))
        if test_i == pred_i:
            correct += 1
        done += 1
        print('Actual', test_i, 'Predicted', pred_i, '(', int((correct / done) * 100), '%)')
        total_time += elapsed

    print('Total Accuracy', correct / done, '(', int((correct / done) * 100), '%)')
    print('Average Time', total_time / done)
