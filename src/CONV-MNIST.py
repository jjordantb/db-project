# Using a Simple Convolution network on MNIST

import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from ImageStore import ImageStore
import ImgUtil

import random

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


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
        size = len(imgs)
        for i, img in enumerate(imgs):
            if i >= size * split:
                self.x_test.append(np.reshape(np.array([x / 255 for x in img]), (28, 28, 1)))  # normalize
                self.y_test.append(np.array(vect))
            else:
                self.x_train.append(np.reshape(np.array([x / 255 for x in img]), (28, 28, 1)))  # normalize
                self.y_train.append(np.array(vect))

    def fetch_random_image(self):
        rand_image = self.images[random.randrange(0, len(self.images))]
        return np.reshape(np.array([x / 255 for x in rand_image]), (28, 28, 1))

    def fetch_random_raw_image(self):
        return self.images[random.randrange(0, len(self.images))]


image_stores = []

# Parameters Start
training_test_split = 0.90
batch_size = 32
num_classes = 10
epochs = 10
# Parameters End

# Init our training data
for i in range(0, 10):
    imgs = ImgUtil.load_mnist(i)
    image_stores.append(MnistStore(i, training_test_split))

x_train = []
y_train = []
x_test = []
y_test = []
for store in image_stores:
    x_train += store.x_train
    y_train += store.y_train
    x_test += store.x_test
    y_test += store.y_test

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
evaluation = model.evaluate(x_test, y_test, verbose=True)
print('Testing, Accuracy=', evaluation[1], 'Loss=', evaluation[0])

print('What MNIST image would you like to make inference on: ')
idx = input()
print('Grabbing random image ' + idx + ' from ImageStore')
results = model.predict(np.array([image_stores[int(idx)].fetch_random_image()]))
label = results[0].argmax(axis=0)
print('The predicted label was', label, 'Here is a random image that is also a', label)
ImgUtil.draw_image(image_stores[int(idx)].fetch_random_raw_image(), 28, 28)
