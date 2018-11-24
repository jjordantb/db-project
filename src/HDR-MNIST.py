import ImgUtil
from Node import Node
import time
import numpy as np
from scipy.spatial import distance

# Parameters Start
training_split_percent = 0.999
nodes = 50
# Parameters End

training = []  # tuple of (image data vector, 1-hot-vector)
testing = []
for i in range(0, 10):
    imgs = ImgUtil.load_mnist(i)
    vect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vect[i] = 1
    size = len(imgs)
    for i, img in enumerate(imgs):
        if i >= size * training_split_percent:
            testing.append((img, vect))
        else:
            training.append((img, vect))


# def print_terminal(n):
#     for i, x_cluster in enumerate(n.x_clusters):
#         if len(x_cluster.child_nodes) > 0:
#             for child in x_cluster.child_nodes:
#                 print_terminal(child)
#         else:
#             print(n.y_clusters[i].mean_vector)
#
#
# print('Training', len(training))
# node = Node(nodes, 10, 1)
# node.build_tree(training, 100)
# print_terminal(node)


def traverse_tree(n, x_in, terminals):
    closest_cluster_pair = n.get_closest_cluster_pair(x_in)
    if len(closest_cluster_pair) > 0:
        for c in closest_cluster_pair:
            if len(c[1].child_nodes) > 0:
                for child in c[1].child_nodes:
                    traverse_tree(child, x_in, terminals)
            else:
                terminals.append(c)


print('Training', len(training))
node = Node(nodes, 10, 1)
node.build_tree(training, 500)
print('Testing', len(testing))
correct = 0
for test in testing:
    print("START ----------------- ")
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
    print(closest[0], closest[2].mean_vector)
    elapsed = int(round(time.time() * 1000)) - start_time
    print('Time', elapsed)
    print("STOP ------------------ ")
    # elapsed = int(round(time.time() * 1000)) - start_time
    # test_i = test[1].index(1)
    # pred_i = pred.index(max(pred))
    # print('Actual ' + str(test_i) + ' Prediction ' + str(pred_i), 'in', elapsed, 'ms')
    # print('Actual ' + str(test[1]) + ' Prediction ' + str(pred), 'in', elapsed, 'ms')
    # if test_i == pred_i:
    #     correct += 1
    # print('Current Accuracy', correct / len(testing), '(', int((correct / len(testing)) * 100), '%)')

print('Total Accuracy', correct / len(testing), '(', int((correct / len(testing)) * 100), '%)')
