from __future__ import print_function
import tensorflow as tf
from keras import losses
import numpy as np
#import h5py

from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, Activation, GaussianNoise, BatchNormalization
from keras import backend as K
from MyConv2D_layer import MyConv2D
from MyDense_layer import MyDense

from l2_attack import CarliniL2
#from l0_attack import CarliniL0
#from li_attack import CarliniLi

from setup_fake_mnist import FakeModel
from Load_data import load_MNIST

from test_attack import generate_data, show
import time

# tf.compat.v1.disable_eager_execution()
# tf.config.run_functions_eagerly(True)


batch_size = 64
num_classes = 10
epochs = 50
train_samples=100#60000
test_samples=10 #10000
validation_samples = 5 #5000
params = [20, 20, 800, 10, 10]

filename_best_sc = 'my_LeNet5_best_SC'
filename_best = 'my_LeNet5_best'
filename_h5 = '.h5'
filename_keras = '.keras'

# tf.compat.v1.disable_eager_execution()

# input image dimensions
img_rows, img_cols = 28, 28



def train_main(data, train_1_test_2, is_SC, params):

    # x_train, y_train, x_test, y_test, input_shape = data

    model = Sequential()
    filepath = filename_best + filename_h5

    if is_SC==True:

        model.add(MyConv2D(filters=params[0], kernel=(5, 5), Method = 'Fixed', Width = 8, Sobol_num1 = 1, 
                           Sobol_num2 = 4, Stream_Length = 16, input_shape=data.input_shape))		#used to be Stoch 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(MyConv2D(filters=params[1], kernel=(5, 5), Method = 'Float', Width = 8, Sobol_num1 = 2, 
                           Sobol_num2 = 4, Stream_Length = 1024))			        #used to be Fixed	
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(MyDense(filters=params[2], Bias=True, Method = 'Float', WidthIn=8, WidthOut=8, Str_Len=16))                                                #used to be Fixed
        model.add(Activation('relu'))
        model.add(MyDense(filters=params[3], Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=256))                                                #used to be Fixed
        model.add(Activation('relu'))
        model.add(MyDense(filters=num_classes, Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=1024))                                                #used to be Fixed
        # model.add(Activation('softmax'))


        #----Saving the best training weights 'my_LeNet5_best_SC.h5'
        # filepath = filename_best_sc + filename_h5        #Save the best epoch
        # savepath = filename_best_sc + filename_keras
    
    elif is_SC==False:

        #-----the NN in binary mode with our new method
        model.add(MyConv2D(filters=params[0], kernel=(5, 5), Method = 'Float', Width = 8, Sobol_num1 = 1, 
                           Sobol_num2 = 4, Stream_Length = 1024, input_shape=data.input_shape))		 
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
        # model.add(Activation('softmax'))


    model.compile(loss=losses.CategoricalCrossentropy(), optimizer='adam' ,#tf.keras.optimizers.Adam(), 
                  metrics=['accuracy'])


    if train_1_test_2==1: ##--only for train
        #----options for monitor: 'val_loss', 'val_accuracy', ...(it seems that 'val_accuracy' works better for SC)
        model_saver_best = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                            save_best_only=True, mode='auto', save_weights_only=True, 
                                                            save_freq='epoch') 
                                                            #options=tf.train.CheckpointOptions()) ,   save_freq=5*batch_size   , period=1
        model.fit(data.x_train, data.y_train,	batch_size=batch_size, epochs=epochs, verbose=1, 
                  validation_data=(data.x_validation, data.y_validation),callbacks=[model_saver_best])
        model.summary()

    
    elif train_1_test_2==2:	  
        #----only for inference
        model.load_weights(filepath)
        print("Weigths are loaded from disk")
        test_loss, test_acc = model.evaluate(data.x_test, data.y_test)
        print("we are in SC:", is_SC)
        print(f'Test accuracy: {test_acc}, Test loss: {test_loss}')


    #----for both train and inference	
    print('batch_size',batch_size,'epochs',epochs)	  
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    #print('Number of Errors:', test_samples - (score[1]*test_samples),'out of ',test_samples)


def attack_main():
    with tf.compat.v1.Session() as sess:
        # data_set = load_MNIST(train_samples=train_samples, test_samples=test_samples, 
        #               validation_samples=validation_samples, img_rows=img_rows, img_cols=img_cols, 
        #               num_classes=num_classes)
        # data_set = load_MNIST_data()
        model_name = filename_best + filename_h5 
        
        # model defenition moved to class: FakeModel
        model_fake = FakeModel(model_name, num_classes, params, sess)
        attack = CarliniL2(sess, model_fake, batch_size=9, max_iterations=1000, confidence=0)
        inputs, targets = generate_data(data_set, samples=1, targeted=True, 
                                        start=0, inception=False)
        print(data_set.y_test.shape)
        print(data_set.y_test.shape[1])
        print(inputs.shape  )
        print(targets)
        print("222222222222222222222222222222222222222222222222222222222222")
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        # adv = attack.attack(data_set.x_test, data_set.y_test)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run ",len(inputs),"samples.")

        print(adv.shape)
        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i])
            # print("Data Set:")
            # show(data_set.x_test[i])
            print("Adversarial:")
            show(adv[i])
            
            print("Classification:", model_fake.model.predict(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)
            # sess.close()

    return adv


#load data once
data_set = load_MNIST(train_samples=train_samples, test_samples=test_samples, 
                      validation_samples=validation_samples, img_rows=img_rows, img_cols=img_cols, 
                      num_classes=num_classes)
# data_set = load_MNIST_data()
print("done 1 : data loaded -----------------------")

#now train the binary Lenet model
train_main(data_set, train_1_test_2=1, is_SC=False, params=params)
print("done 3 : nn trained -----------------------")

# #test the stochastic model
# train_main(data_set, train_1_test_2=2, is_SC=True, params=params)
# print("done 4 : SC tested -----------------------")

#test binary model now
# train_main(data_set, train_1_test_2=2, is_SC=False, params=params)
# print("done 5 : nn tested -----------------------")


#apply attack
adv_test_data = attack_main()
print("done 6 : attack applied -----------------------")


# #test SC network, after attack being applied
# train_main((data_set[0], data_set[1], adv_test_data, data_set[3], data_set[4]), 
#            train_1_test_2=2, is_SC=True, params=params)
# print("done 7 : SC tested after attack -----------------------")


#test binary network, after attack being applied
# train_main((data_set[0], data_set[1], adv_test_data, adv_test_labels, shape_), 
#            train_1_test_2=2, is_SC=False, params=params)
# print("done 8 : SC tested after attack -----------------------")
