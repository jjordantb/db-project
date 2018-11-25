# Using a Simple Convolution network on MNIST
import time
import os
import sys

import keras
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from ImageStore import ImageStore
import ImgUtil

from PIL import Image, ImageDraw, ImageOps
import PIL
from tkinter import *

import random

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force cpu

# Parameters Start
training_test_split = 0.95
batch_size = 32
num_classes = 10
epochs = 5
# Parameters End


def current_time_ms():
    return int(round(time.time() * 1000))


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

# Init our training data
for i in range(0, 10):
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

print('Training...')
start_train = current_time_ms()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
print('Training took', current_time_ms() - start_train)
evaluation = model.evaluate(x_test, y_test, verbose=True)
print('Testing, Accuracy=', evaluation[1], 'Loss=', evaluation[0])

if 'demo' in sys.argv:
    root = Tk()


    def on_quit():
        quit(0)


    width = 128
    height = 128
    center = height // 2
    white = (255, 255, 255)

    root.protocol('WM_DELETE_WINDOW', on_quit)


    def on_button():
        im = ImageOps.invert(image1)
        im.thumbnail((28, 28))
        im.save('cov.png')
        raw = []
        for pixel in iter(im.getdata()):
            raw.append(pixel[0])
        results = model.predict(np.array([np.array([x / 255 for x in raw]).reshape(28, 28, 1)]))
        label = results[0].argmax(axis=0)
        print('The predicted label was', label, 'Here is a random image that is also a', label)
        store_image = image_stores[int(label)].fetch_random_raw_image()
        ImgUtil.draw_image(store_image, 28, 28)
        cv.delete('all')
        image1.putdata(np.zeros((128, 128), dtype='i,i,i'))


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
