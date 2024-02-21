# import keras
import tensorflow as tf

from keras.datasets import mnist, cifar10, cifar100
from keras import backend as K
import numpy as np
import pickle

#===================================================================================
def DataPreparation(DataSet, train_samples, test_samples, num_classes, subtract_pixel_mean):
    #-------------------------------------------------------
    if DataSet == 'MNIST':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        input_shape = [x_train.shape[1], x_train.shape[2], 1]
    elif DataSet == 'CIFAR10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        input_shape = x_train.shape[1:]
    elif DataSet == 'CIFAR100':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        input_shape = x_train.shape[1:]
    elif DataSet == 'ImageNet':
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        input_shape = x_train.shape[1:]

    #-------------------------------------------------------
    x_train = x_train[0:train_samples]
    y_train = y_train[0:train_samples]
    x_test  = x_test[0:test_samples]
    y_test  = y_test[0:test_samples]

    #-------------------------------------------------------
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, input_shape[0], input_shape[1])
        x_test = x_test.reshape(x_test.shape[0], 1, input_shape[0], input_shape[1])
    else:
        x_train = x_train.reshape(x_train.shape[0], input_shape[0], input_shape[1], input_shape[2])
        x_test = x_test.reshape(x_test.shape[0], input_shape[0], input_shape[1], input_shape[2])
        
    #-------------------------------------------------------
    # Normalize data and put it in [-0.5;0.5] range
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train -= 0.5
    x_test -= 0.5
    print( "__________ minus happened __________")

    #-------------------------------------------------------
    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    #-------------------------------------------------------
    # Display the shapes to check if everything's ok
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print('input_shape:', input_shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    #-------------------------------------------------------
    # Convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test  = tf.keras.utils.to_categorical(y_test, num_classes)
    
    #-------------------------------------------------------
    return ((x_train, y_train), (x_test, y_test), input_shape) 	
#===================================================================================
def AfterAdvDataPreparation(y):

    with open('adv_attack_x.pkl', 'rb') as file:
        x_attacked = pickle.load(file)
    # with open('adv_attack_y.pkl', 'rb') as file:
    #     y_attacked = pickle.load(file)
            
    x_test = x_attacked
    yy = []
    for i in range(y.shape[0]):
        for j in range(0,9):
            yy.append(y[i])
            j+=1
    y_test = np.array(yy[:x_attacked.shape[0]])
    print(y.shape)
    print(y_test.shape)
    return (x_test, y_test)
#===================================================================================
