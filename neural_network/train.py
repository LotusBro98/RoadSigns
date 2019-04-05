#!/usr/bin/python3

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import subprocess

import cv2 as cv

import time

CLASSES_PATH = "../dataset/classes"

PATTERN_WIDTH = 32
PATTERN_HEIGHT = 32

epochs = 100

train_images = []
train_labels = []

# def grey_world(nimg):
#     nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
#     mu_g = np.average(nimg[1])
#     nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
#     nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
#     return  nimg.transpose(1, 2, 0).astype(np.uint8)
#
# def apply_clahe(img):
#     clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#
#     blue, green, red = cv.split(img)
#
#     red_clahe = clahe.apply(red)
#     green_clahe = clahe.apply(green)
#     blue_clahe = clahe.apply(blue)
#
#     return cv.merge((blue_clahe, green_clahe, red_clahe))


n_classes = len(os.listdir(CLASSES_PATH))

def load_image(filename, label): 
    image = cv.imread(filename)
#    image = apply_clahe(image)
    image = cv.resize(image, (PATTERN_WIDTH, PATTERN_HEIGHT))
#    image = grey_world(image)
    image = image / 255.0
    train_images.append(image)

    label_arr = np.zeros(n_classes)
    label_arr[label] = 1
    train_labels.append(label_arr)


label = 0
for classname in os.listdir(CLASSES_PATH):
    for filename in os.listdir(os.path.join(CLASSES_PATH, classname)):
        load_image(os.path.join(CLASSES_PATH, classname, filename), label)
    label += 1



X_train = np.asarray(train_images)
Y_train = np.asarray(train_labels)

batch_size = int(Y_train.shape[0])

model = tf.keras.Sequential()

model.add(tf.keras.layers.Convolution2D(32, (3, 3), input_shape=(PATTERN_WIDTH, PATTERN_HEIGHT, 3), activation='relu'))
model.add(tf.keras.layers.Convolution2D(32, (3, 3), activation='relu'))  
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))   
 
model.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu')) 
model.add(tf.keras.layers.Convolution2D(64, (3, 3), activation='relu'))     
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))     
model.add(tf.keras.layers.Dropout(0.25))
                
model.add(tf.keras.layers.Flatten())
  
model.add(tf.keras.layers.Dense(200, activation='relu')) 
model.add(tf.keras.layers.Dropout(0.5))      
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))

model.compile(loss=tf.losses.mean_squared_error, 
    metrics=['accuracy'], optimizer='adam')


model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=True) 

print('inputs: ', [input.op.name for input in model.inputs])

print('outputs: ', [output.op.name for output in model.outputs])

model.save("model.h5", overwrite=True, include_optimizer=False)

os.system("python keras_to_tensorflow/keras_to_tensorflow.py --input_model=\"./model.h5\" --output_model=\"./model.pb\"")

####################################################
