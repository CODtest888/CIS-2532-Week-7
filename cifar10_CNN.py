#########################################################################
#Application Name: cifar10_CNN.py
#Description:      Creates a model to classify pictures
#Author:           josephlee94 / intuitive-deep-learning / GitHub
#Comments:         Ryan Buechler
#Course Name:      CIS-2532
#Section:          NET01
#Instructor:       Mohammad Morovati
#Assignment#:      Assignment #11
#Date:             08/03/2025
#########################################################################


# Downloads the cifar10 dataset and splits it into training and testing sets
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print('x_train shape:', x_train.shape)

print('y_train shape:', y_train.shape)

print(x_train[0])  # Displays one of the images as numbers

import matplotlib.pyplot as plt
#%matplotlib inline

# Converts and displays images so that they display as images instead of numbers
img = plt.imshow(x_train[0])

print('The label is:', y_train[0])

img = plt.imshow(x_train[1])

print('The label is:', y_train[1])

import keras

# One out of ten images is set aside to calculate the probability of being in a certain class
y_train_one_hot = keras.utils.to_categorical(y_train, 10)
y_test_one_hot = keras.utils.to_categorical(y_test, 10)

print('The one hot label is:', y_train_one_hot[1])

# Converts the numbers representing each picture to be between 0 and 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

print(x_train[0])

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Uses a sequential model
model = Sequential()

# Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32,32,3)))

# Layer 2
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

# Layer 3
model.add(MaxPooling2D(pool_size=(2, 2)))

# Droput layer
model.add(Dropout(0.25))

# Repeats the four layers with a conv layer of 64 instead of 32
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Converts the neurons from a cube-like format to a row
model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

model.summary()


# Compiles the training model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Trains the model
hist = model.fit(x_train, y_train_one_hot, 
           batch_size=32, epochs=20, 
           validation_split=0.2)

# Displays the model loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Displays the  model accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Runs the traing set against the test set
model.evaluate(x_test, y_test_one_hot)[1]

# Saves the model
model.save('my_cifar10_model.h5')

# Imports an image as pixel values
my_image = plt.imread("cat.jpg")

from skimage.transform import resize

# Resizes the the image
my_image_resized = resize(my_image, (32,32,3))

# Displays the image as an image
img = plt.imshow(my_image_resized)

import numpy as np

# Calculates the probability of the image being in each class
probabilities = model.predict(np.array( [my_image_resized,] ))

print(probabilities)

number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0,:])
print("Most likely class:", number_to_class[index[9]], "-- Probability:", probabilities[0,index[9]])
print("Second most likely class:", number_to_class[index[8]], "-- Probability:", probabilities[0,index[8]])
print("Third most likely class:", number_to_class[index[7]], "-- Probability:", probabilities[0,index[7]])
print("Fourth most likely class:", number_to_class[index[6]], "-- Probability:", probabilities[0,index[6]])
print("Fifth most likely class:", number_to_class[index[5]], "-- Probability:", probabilities[0,index[5]])
