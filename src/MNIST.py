import ImgUtil
from Node import Node
import time
import numpy as np

# Parameters Start
training_split_percent = 0.8
# Parameters End

training = []  # tuple of (image data vector, 1-hot-vector)
testing = []
for i in range(0, 10):
    imgs = ImgUtil.load_mnist(i)
    vect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vect[i] = 1
    size = len(imgs)
    for i, img in enumerate(imgs):
        if i > size * training_split_percent:
            testing.append((img, vect))
        else:
            training.append((img, vect))


def traverse_tree(n, x_in, cur):
    closest_cluster_pair = n.get_closest_cluster_pair(x_in)
    if cur is None or cur[0] > closest_cluster_pair[0]:
        cur = closest_cluster_pair
    if len(closest_cluster_pair[1].child_nodes) > 0:
        for child in closest_cluster_pair[1].child_nodes:
            return traverse_tree(child, x_in, cur)
    return closest_cluster_pair


print('Training', len(training))
node = Node(10, 10)
node.build_tree(training, 1850)
print('Testing', len(testing))
correct = 0
for test in testing:
    start_time = int(round(time.time() * 1000))
    pred = traverse_tree(node, test[0], None)[2].mean_vector
    elapsed = int(round(time.time() * 1000)) - start_time
    print('Actual ' + str(test[1]) + ' Prediction ' + str(pred), 'in', elapsed, 'ms')
    if np.array_equal(np.array(test[1]), np.array(pred)):
        correct += 1

print('Total Accuracy', correct / len(testing), '(', int((correct / len(testing)) * 100), '%)')
