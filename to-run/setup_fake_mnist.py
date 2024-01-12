## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Activation, Flatten
from keras.layers import MaxPooling2D

from MyConv2D_layer import MyConv2D
from MyDense_layer import MyDense



import gzip



def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data / 255) - 0.5
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)


class FakeModel:
    def __init__(self, restore, num_classes, params, session=None): #batch_size=9, 
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        model.add(MyConv2D(filters=params[0], kernel=(5, 5), Method = 'Float', Width = 8, Sobol_num1 = 1, 
                        Sobol_num2 = 4, Stream_Length = 1024, input_shape=(28, 28, 1)))		 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(MyConv2D(filters=params[1], kernel=(5, 5), Method = 'Float', Width = 8, Sobol_num1 = 2, 
                        Sobol_num2 = 4, Stream_Length = 1024))			       	
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(MyDense(filters=params[2], Bias=True, Method = 'Float', WidthIn=8, WidthOut=8, Str_Len=256))                           
        model.add(Activation('relu'))
        model.add(MyDense(filters=params[3], Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=256))                           
        model.add(Activation('relu'))
        model.add(MyDense(filters=num_classes, Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=1024))
  
        # model.build((batch_size, 28, 28, 1)) #(batch_size, img_rows, img_cols, num_channels))  # Replace with your model's input shape, None is for batch size
        
        print("Weigths are NOOOOT loaded from disk") 
        model.load_weights(restore)
        print("Weigths are loaded from disk")

        # model.summary()
        self.model = model

    def predict(self, data):
        # p = tf.constant([[1, 2, 3], [4, 5, 6]])
        # print(data.shape)
        # print(data)
        # print("++++++++++++++++++++++++++++++++++++")
        out = self.model(data)
        return out
    

# import os
# import pickle
# import gzip
# import urllib.request

# def extract_data(filename, num_images):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(16)
#         buf = bytestream.read(num_images*28*28)
#         data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
#         data = (data / 255) - 0.5
#         data = data.reshape(num_images, 28, 28, 1)
#         return data

# def extract_labels(filename, num_images):
#     with gzip.open(filename) as bytestream:
#         bytestream.read(8)
#         buf = bytestream.read(1 * num_images)
#         labels = np.frombuffer(buf, dtype=np.uint8)
#     return (np.arange(10) == labels[:, None]).astype(np.float32)

# class MNIST:
#     def __init__(self):
#         if not os.path.exists("data"):
#             os.mkdir("data")
#             files = ["train-images-idx3-ubyte.gz",
#                      "t10k-images-idx3-ubyte.gz",
#                      "train-labels-idx1-ubyte.gz",
#                      "t10k-labels-idx1-ubyte.gz"]
#             for name in files:

#                 urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

#         train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
#         train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
#         self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
#         self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)
        
#         VALIDATION_SIZE = 5000
        
#         self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
#         self.validation_labels = train_labels[:VALIDATION_SIZE]
#         self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
#         self.train_labels = train_labels[VALIDATION_SIZE:]

