from __future__ import print_function
import tensorflow as tf
from keras import losses
import numpy as np
#import h5py
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, Activation, GaussianNoise, BatchNormalization
from keras import backend as K
from MyConv2D_layer import MyConv2D
from MyDense_layer import MyDense

from l2_attack import CarliniL2
#from l0_attack import CarliniL0
#from li_attack import CarliniLi

import test_attack
import time

batch_size = 64
num_classes = 10
epochs = 50
train_samples=500 #60000
test_samples=100 #10000
params = [20, 20, 800, 10, 10]

filename_best_sc = 'my_LeNet5_best_SC'
filename_best = 'my_LeNet5_best'
filename_h5 = '.h5'
filename_keras = '.keras'

#train_1_test_2 = 1



# input image dimensions
img_rows, img_cols = 28, 28


def introduce_random_errors(images, error_rate=0.05):    
    noisy_images = images.copy()
    for img in noisy_images:        # Number of pixels to alter
        num_errors = int(error_rate * img.size)        
        for _ in range(num_errors):
            # Randomly choose a pixel and change its value            
            x, y = np.random.randint(0, img.shape[0]), np.random.randint(0, img.shape[1])
            img[x, y] = np.random.randint(0, 256)    
    return noisy_images
        

def load_MNIST_data(error=False):
    
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # select a subset of the dataset for the train and test
    x_train = x_train[0:train_samples];y_train = y_train[0:train_samples]
    #x_test = x_test[0:5000];y_test = y_test[0:5000];
    x_test = x_test[0:test_samples];y_test = y_test[0:test_samples]

    # input_shape = x_train.shape[1:]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    if error == True:
        # add random error to the input training data 
        x_train = introduce_random_errors(x_train)

    # Make the value floats in [0;1] instead of int in [0;255]	
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    #Display the shapes to check if everything's ok
    print('x_train shape:', x_train.shape)
    print('y_train shape:', y_train.shape)
    print('x_test shape:', x_test.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train, x_test, y_test, input_shape)


def train_main(data, train_1_test_2, is_SC, params):

    x_train, y_train, x_test, y_test, input_shape = data

    model = Sequential()

    if is_SC==True:

        model.add(MyConv2D(filters=params[0], kernel=(5, 5), Method = 'Stoch', Width = 8, Sobol_num1 = 1, 
                           Sobol_num2 = 4, Stream_Length = 1024, input_shape=input_shape))		#used to be Stoch 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(MyConv2D(filters=params[1], kernel=(5, 5), Method = 'Stoch', Width = 8, Sobol_num1 = 2, 
                           Sobol_num2 = 4, Stream_Length = 1024))			        #used to be Fixed	
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(MyDense(filters=params[2], Bias=True, Method = 'Float', WidthIn=8, WidthOut=8, Str_Len=256))                                                #used to be Fixed
        model.add(Activation('relu'))
        model.add(MyDense(filters=params[3], Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=256))                                                #used to be Fixed
        model.add(Activation('relu'))
        model.add(MyDense(filters=num_classes, Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=1024))                                                #used to be Fixed
        model.add(Activation('softmax'))


        #----Saving the best training weights 'my_LeNet5_best_SC.h5'
        filepath = filename_best_sc + filename_h5        #Save the best epoch
        savepath = filename_best_sc + filename_keras
    
    elif is_SC==False:

        model.add(Conv2D(params[0], (5, 5), activation='relu', use_bias=True, bias_initializer='RandomNormal', 
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(params[1], (5, 5), activation='relu', use_bias=True, bias_initializer='RandomNormal'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(params[2], input_dim=320, activation='relu', use_bias=True, bias_initializer='RandomNormal'))
        model.add(Dense(params[3], input_dim=800, activation='relu', use_bias=True, bias_initializer='RandomNormal'))
        model.add(Dense(num_classes, activation='softmax', use_bias=True, bias_initializer='RandomNormal'))

        #----Saving the best training weights 'my_LeNet5_best_.h5'
        filepath = filename_best + filename_h5             #Save the best epoch
        savepath = filename_best + filename_keras


    model.compile(loss=losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])


    if train_1_test_2==1: ##--only for train
        ##----Saving the best training weights 'my_LeNet5_best.h5'
        #filepath = 'my_LeNet5_best.h5'; #Save the best epoch

        #----options for monitor: 'val_loss', 'val_accuracy', ...(it seems that 'val_accuracy' works better for SC)
        model_saver_best = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                            save_best_only=True, mode='auto', save_weights_only=True, 
                                                            save_freq='epoch') 
                                                            #options=tf.train.CheckpointOptions()) ,   save_freq=5*batch_size   , period=1
        model.fit(x_train, y_train,	batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),callbacks=[model_saver_best])
        model.summary()
        # model.save(savepath)

            
    
    elif train_1_test_2==2:	  
        #----only for inference
        model.load_weights(filepath)
        print("Weigths are loaded from disk")
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print("we are in SC:", is_SC)
        print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')


    #----for both train and inference	
    print('batch_size',batch_size,'epochs',epochs)	  
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    #print('Number of Errors:', test_samples - (score[1]*test_samples),'out of ',test_samples)


def attack_main(input_shape, test_data, test_labels, batch_size=9, max_iterations=1000, confidence=0, is_SC=True):
    num_channels = 1
    image_size = img_rows       #28
    num_labels = num_classes    #10
    
    
    if is_SC:
        model_name = filename_best_sc + filename_keras
    else:
        model_name = filename_best + filename_h5 # filename_keras

    # with tf.compat.v1.Session() as sess:

        #loaded_model = tf.keras.saving.load_model(model_name, compile=True)
    model_name = filename_best + filename_keras
    model = Sequential()

    model.add(Conv2D(params[0], (5, 5), activation='relu', use_bias=True, bias_initializer='RandomNormal', 
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(params[1], (5, 5), activation='relu', use_bias=True, bias_initializer='RandomNormal'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(params[2], input_dim=320, activation='relu', use_bias=True, bias_initializer='RandomNormal'))
    model.add(Dense(params[3], input_dim=800, activation='relu', use_bias=True, bias_initializer='RandomNormal'))
    model.add(Dense(num_classes, activation='softmax', use_bias=True, bias_initializer='RandomNormal'))

    #model.compile(loss=losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
    model.build((batch_size, img_rows, img_cols, num_channels))  # Replace with your model's input shape, None is for batch size
    
    print("Weigths are NOOOOT loaded from disk") 
    model.load_weights(model_name)
    print("Weigths are loaded from disk")

    model.summary()

    with tf.compat.v1.Session() as sess:

        attack = CarliniL2(sess, model, image_size, num_channels, num_labels,
                           batch_size, max_iterations, confidence)

        inputs, targets = test_attack.generate_data(test_data, test_labels, samples=1, targeted=True, #true
                                        start=0, inception=False)

        # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        # print((inputs.shape))
        # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        for i in range(len(adv)):
            print("Valid:")
            test_attack.show(inputs[i])
            print("Adversarial:")
            test_attack.show(adv[i])
            
            print("Classification:", model(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)


#load data once
data_set = load_MNIST_data(error=True)
print("done 1 : data loaded -----------------------")

#first train the stochastic model
train_main(data_set, train_1_test_2=1, is_SC=True, params=params)
print("done 2 : SC trained -----------------------")

#now train the binary Lenet model
# train_main(data_set, train_1_test_2=1, is_SC=False, params=params)
# print("done 3 : nn trained -----------------------")

#test the stochastic model
train_main(data_set, train_1_test_2=2, is_SC=True, params=params)

#test binary model now
#train_main(data_set, train_1_test_2=2, is_SC=False, params=params)


# #apply attack
# _, _, test_data, test_labels, shape_= data_set
# attack_main(shape_, test_data, test_labels, is_SC=False)

