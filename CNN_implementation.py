import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models,optimizers
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization,Dense, Activation
from keras.models import Model,Sequential
from tensorflow.keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler
import math
import matplotlib.pyplot as plt
print(tf.__version__)
print(keras.__version__)

# fashion_mnist dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, Y_train_full), (X_test, Y_test) = fashion_mnist.load_data()

# Max normalization and seperating Validation set
X_valid, X_train = X_train_full[:6000]/255.0 , X_train_full[6000:]/255.0
Y_valid, Y_train = Y_train_full[:6000] , Y_train_full[6000:]
X_test = X_test / 255.0

Y_train = Y_train.reshape(-1,)
Y_test = Y_test.reshape(-1,)

class_names = ["T-shirt/top", "Trouser", "pullover", "Dress", "Coat", 
				"Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"]

print("X Training data shape - {}".format(X_train.shape))
print("Y Training data type - {}".format(Y_train.shape))

"""Sample of Training Set"""

plt.figure(figsize=(5, 5))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i].reshape((28,28)))
    label_index = int(Y_train[i])
    plt.title(class_names[label_index])
plt.show()

image_rows,image_cols = 28,28
#batch_size = 4096
image_shape = (image_rows,image_cols,1) 
X_train = X_train.reshape(X_train.shape[0],*image_shape)
X_test = X_test.reshape(X_test.shape[0],*image_shape)
X_valid = X_valid.reshape(X_valid.shape[0],*image_shape)

"""**1) Building a CNN Model as per Question 2, Part 1**"""

cnn_model = models.Sequential([                         
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn_model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn_model.fit(X_train, Y_train, epochs=15,validation_data=(X_valid,Y_valid))

"""Evaluating Accuracy of Model 1"""

loss, acc = cnn_model.evaluate(X_test, Y_test)
print('Accuracy of CNN Model is: %.3f' % (acc * 100.0))

"""**2) Adding Image Augmentation on Model 1**"""

batch_size = 64
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(X_train, Y_train, batch_size)
steps_per_epoch = X_train.shape[0] // batch_size
r = cnn_model.fit_generator(train_generator, validation_data=(X_test, Y_test), steps_per_epoch=steps_per_epoch, epochs=10)

loss, acc = cnn_model.evaluate(X_test, Y_test)
print('Accuracy of Cnn Model after augmentation is: %.3f' % (acc * 100.0))

"""**3) Building CNN model with Batch Normalization as per Q2 part 3**"""

cnn_model_withBatch = models.Sequential([                         
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

cnn_model_withBatch.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn_model_withBatch.fit(X_train, Y_train, epochs=10,validation_data=(X_valid,Y_valid))

"""Evaluating Accuracy of CNN MODEL with Batch Normalization"""

loss, acc = cnn_model_withBatch.evaluate(X_test, Y_test)
print('Accuracy of CNN Model with Batch Normalization is: %.3f' % (acc * 100.0))

"""Augmenting this Batch Normalized CNN Model"""

batch_size = 64
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
train_generator = data_generator.flow(X_train, Y_train, batch_size)
steps_per_epoch = X_train.shape[0] // batch_size
r = cnn_model_withBatch.fit_generator(train_generator, validation_data=(X_test, Y_test), steps_per_epoch=steps_per_epoch, epochs=10)

"""Evaluating Accuracy of CNN MODEL with Batch Normalization and Augmentation"""

loss, acc = cnn_model_withBatch.evaluate(X_test, Y_test)
print('Accuracy of CNN Model with Batch Normalization and augmentation is: %.3f' % (acc * 100.0))

"""**4) Using learning Rate scheduler of step decay with SGD optimizer**

Creating Learning Rate Scheduler and fitting the model
"""

# define SGD optimizer
momentum = 0.5
sgd = SGD(lr=0.0, momentum=momentum, decay=0.0, nesterov=False) 

# define step decay function
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
    return lrate

# compile the model
cnn_model_withBatch.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=sgd, metrics=['accuracy'])

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# fit the model
history = cnn_model_withBatch.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=10, batch_size=64, callbacks=callbacks_list, verbose=2)

"""Evaluating the model"""

loss, acc = cnn_model_withBatch.evaluate(X_test,Y_test)
print('Accuracy of CNN Model with Batch Normalization and Step Decay Learning Rate Scheduler is: %.3f' % (acc * 100.0))