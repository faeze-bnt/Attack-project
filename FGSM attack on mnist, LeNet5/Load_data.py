## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import pickle

from keras.datasets import mnist
# from keras.datasets import fashion_mnist
from keras import backend as K


class Load_DATA:
    def __init__(self, error=False, train_samples=60000, test_samples=10000, validation_samples=5000, 
                 img_rows=28, img_cols=28, num_classes=10):
        # the data, split between train and test sets
        # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train[0:train_samples]
        y_train = y_train[0:train_samples]
        x_test = x_test[0:test_samples]
        y_test = y_test[0:test_samples]

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        # add random error to the input training data 
        if error == True:
            x_train = self.introduce_random_errors(x_train)

        # Make the value floats in [0;1] instead of int in [0;255]	
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        x_train -= 0.5
        x_test -= 0.5

        #Display the shapes to check if everything's ok
        print('x_train shape:', x_train.shape)
        print('y_train shape:', y_train.shape)
        print('x_test shape:', x_test.shape)
        print('y_test shape:', y_test.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
 
        self.x_validation = x_train[:validation_samples, :, :, :]
        self.y_validation = y_train[:validation_samples]
        self.x_train = x_train[validation_samples:, :, :, :]
        self.y_train = y_train[validation_samples:]
        self.x_test = x_test
        self.y_test = y_test
        self.input_shape = input_shape
        


    def data_after_attack(self, x, y):
        self.x_test = x
        yy = [] #np.ndarray()#shape=(y.shape[0]*9,y.shape[1]), dtype='float32')
        for i in range(y.shape[0]):
            for j in range(0,9):
                yy.append(y[i])
                j+=1
        self.y_test = np.array(yy[:x.shape[0]])
        print(y.shape)
        # print(yy.shape)
        print(self.y_test.shape)


    def save_after_attack(self, x_attacked, y_attacked):
        with open('adv_attack_x.pkl', 'wb') as file:
            pickle.dump(x_attacked, file)
        # with open('adv_attack_y.pkl', 'wb') as file:
        #     pickle.dump(y_attacked, file)


    def load_after_attack(self):
        with open('adv_attack_x.pkl', 'rb') as file:
            x_attacked = pickle.load(file)
        # with open('adv_attack_y.pkl', 'rb') as file:
        #     y_attacked = pickle.load(file)

        self.data_after_attack(x_attacked, self.y_test)


    def introduce_random_errors(images, error_rate=0.05):    
        noisy_images = images.copy()
        for img in noisy_images:        # Number of pixels to alter
            num_errors = int(error_rate * img.size)        
            for _ in range(num_errors):
                # Randomly choose a pixel and change its value            
                x, y = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
                img[x, y] = np.random.randint(0, 256)    
        return noisy_images