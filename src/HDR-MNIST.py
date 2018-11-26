import time
from tkinter import *

import PIL
import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial import distance
import matplotlib.pyplot as plt

from Node import Node

# Parameters Start
clusters = 50
selectivity = 850
num_classes = 10
k = 10
# Parameters End


def current_time_ms():
    return int(round(time.time() * 1000))


print('Preparing Data')
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('Shape', x_train.shape)


def traverse_tree(node_n, x_in, terminals):
    closest_cluster_pairs = node_n.get_closest_cluster_pairs(x_in, k)
    if len(closest_cluster_pairs) > 0:
        for c in closest_cluster_pairs:
            if len(c[1].child_nodes) > 0:
                for child in c[1].child_nodes:
                    traverse_tree(child, x_in, terminals)
            else:
                terminals.append(c)


start_training = current_time_ms()
print('Training on', len(x_train), 'samples')
node = Node(clusters, 10, 1)
node.build_tree(x_train, y_train, selectivity)
end_training = current_time_ms()
print('Training took', end_training - start_training, 'ms')

if 'demo' in sys.argv:
    root = Tk()


    def on_quit():
        quit(0)


    width = 128
    height = 128
    center = height // 2
    white = (255, 255, 255)
    data = PIL.Image.new("RGB", (width, height), white)

    root.protocol('WM_DELETE_WINDOW', on_quit)


    def on_button():
        im = ImageOps.invert(image1)
        im.thumbnail((28, 28))
        pixels = list(im.getdata())
        width, height = im.size
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
        raw = []
        for i in pixels:
            for j in i:
                raw.append(j[0])
        results = []
        traverse_tree(node, raw, results)
        closest = None
        for result in results:
            dist = distance.euclidean(np.array(result[1].mean_vector), raw)
            if closest is None or closest[0] > dist:
                result[0] = dist
                closest = result

        pred = closest[2].mean_vector
        pred_i = pred.index(max(pred)) if isinstance(pred, list) else pred.tolist().index(max(pred))
        print('The predicted label was', pred_i, 'Here is a random image that is also a', pred_i)
        plt.subplot(3, 1, 1)
        plt.title('Mean Vector of Terminal Node')
        plt.imshow(closest[1].mean_vector.reshape((28, 28)), cmap='gray')
        plt.subplot(3, 1, 2)
        plt.title('XVector[0] of Terminal Node')
        plt.imshow(closest[1].x_vectors[0].reshape((28, 28)), cmap='gray')
        plt.subplot(3, 1, 3)
        plt.title('Is this a ' + str(pred_i))
        plt.imshow(im, cmap='gray')
        plt.show()
        cv.delete('all')
        image1.putdata(data.getdata())


    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        cv.create_oval(x1, y1, x2, y2, fill="black", width=6)
        draw.line([x1, y1, x2, y2], fill="black", width=6)


    while True:
        cv = Canvas(root, width=width, height=height, bg='white')
        cv.pack()
        image1 = PIL.Image.new("RGB", (width, height), white)
        draw = ImageDraw.Draw(image1)
        cv.pack(expand=YES, fill=BOTH)
        cv.bind("<B1-Motion>", paint)
        button = Button(text="query", command=on_button)
        button.pack()
        root.mainloop()

else:
    print('Testing', len(x_test), 'data points')
    correct = 0
    done = 0
    total_time = 0
    for i, x in enumerate(x_test):
        y = y_test[i]
        start_time = int(round(time.time() * 1000))
        results = []
        traverse_tree(node, x, results)
        closest = None
        for result in results:
            dist = distance.euclidean(np.array(result[1].mean_vector), x)
            if closest is None or closest[0] > dist:
                result[0] = dist
                closest = result

        elapsed = int(round(time.time() * 1000)) - start_time
        pred = closest[2].mean_vector
        test_i = y.tolist().index(1)
        pred_i = pred.index(max(pred)) if isinstance(pred, list) else pred.tolist().index(max(pred))
        if test_i == pred_i:
            correct += 1
        done += 1
        print('Actual', test_i, 'Predicted', pred_i, '(', int((correct / done) * 100), '%)', i)
        total_time += elapsed

    print('Total Accuracy', correct / done, '(', int((correct / done) * 100), '%)')
    print('Average Time', total_time / done)
    print(end_training - start_training)
