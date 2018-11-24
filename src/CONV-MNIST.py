# Using a Simple Convolution network on MNIST

import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

import ImgUtil

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


# Parameters Start
training_test_split = 0.90
batch_size = 32
num_classes = 10
epochs = 20
# Parameters End

# tuple of (image data vector, 1-hot-vector)
training = []
testing = []
for i in range(0, 10):
    imgs = ImgUtil.load_mnist(i)
    vect = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    vect[i] = 1
    size = len(imgs)
    for i, img in enumerate(imgs):
        if i >= size * training_test_split:
            testing.append(([x / 255 for x in img], vect))  # normalize
        else:
            training.append(([x / 255 for x in img], vect))  # normalize

# reshape data for network
x_train = []
y_train = []
for item in training:
    x_train.append(np.reshape(np.array(item[0]), (28, 28, 1)))
    y_train.append(np.array(item[1]))
x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []
for item in testing:
    x_test.append(np.reshape(np.array(item[0]), (28, 28, 1)))
    y_test.append(np.array(item[1]))
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)
print(y_train.shape)

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
