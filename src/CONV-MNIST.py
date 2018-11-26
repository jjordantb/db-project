# Using a Simple Convolution network on MNIST
import time
from tkinter import *

import PIL
import keras
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force cpu

# Parameters Start
batch_size = 256
num_classes = 10
epochs = 10
# Parameters End


def current_time_ms():
    return int(round(time.time() * 1000))


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
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
    data = PIL.Image.new("RGB", (width, height), white)

    root.protocol('WM_DELETE_WINDOW', on_quit)


    def on_button():
        im = ImageOps.invert(image1)
        im.thumbnail((28, 28))
        raw = []
        for pixel in iter(im.getdata()):
            raw.append(pixel[0])
        results = model.predict(np.array([np.array([x / 255 for x in raw]).reshape(28, 28, 1)]))
        label = results[0].argmax(axis=0)
        print('The predicted label was', label, 'Here is a random image that is also a', label)
        plt.title('Is this a ' + str(label))
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
