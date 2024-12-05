from __future__ import print_function
import torch
import torch.nn.functional as F
import tensorflow as tf
from keras import losses
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, Activation, GaussianNoise, BatchNormalization, Input
from keras import backend as K
from MyConv2D_layer import MyConv2D
from MyDense_layer import MyDense

# from l2_attack import CarliniL2
#from l0_attack import CarliniL0
#from li_attack import CarliniLi

from setup_fake_mnist import FakeModel
from Load_data import Load_DATA
from Confusion_matrix import Conf_matrix

from test_attack import generate_data, show
import time


# from autoattack import AutoAttack
# import utils_tf2

epsilon = 0.3
batch_size = 64
num_classes = 10
epochs = 50 
train_samples= 60000
test_samples= 500
validation_samples = 500
params = [20, 20, 800, 10, 10]

filename_best_sc = 'my_LeNet5_best_SC'
filename_best = 'my_LeNet5_best'
filename_h5 = '.weights.h5'
filename_keras = '.keras'

# input image dimensions
img_rows, img_cols = 28, 28



def train_main(data, train_1_test_2, is_SC, params, is_attack=False):

    model = Sequential()
    filepath = filename_best + filename_h5

    if is_SC==True:
        # Method could be: Stoch - Fixed - Float
        model.add(MyConv2D(filters=params[0], kernel=(5, 5), Method = 'Float', Width = 8, Sobol_num1 = 1,       # 1 , 4
                           Sobol_num2 = 4, Stream_Length = 8, input_shape=data.input_shape))	 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(MyConv2D(filters=params[1], kernel=(5, 5), Method = 'Stoch', Width = 8, Sobol_num1 = 1,       # 2 , 4
                           Sobol_num2 = 4, Stream_Length = 512))		
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(MyDense(filters=params[2], Bias=True, Method = 'Float', WidthIn=8, WidthOut=8, Str_Len=16))                                              
        model.add(Activation('relu'))
        model.add(MyDense(filters=params[3], Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=256))                                              
        model.add(Activation('relu'))
        model.add(MyDense(filters=num_classes, Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=1024))                                             
        model.add(Activation('softmax'))


    elif is_SC==False:

        #-----the NN in binary mode with our new method
        model.add(MyConv2D(filters=params[0], kernel=(5, 5), Method = 'Float', Width = 8, Sobol_num1 = 1, 
                           Sobol_num2 = 4, Stream_Length = 1024, input_shape=data.input_shape))		 
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(MyConv2D(filters=params[1], kernel=(5, 5), Method = 'Float', Width = 8, Sobol_num1 = 2, 
                           Sobol_num2 = 4, Stream_Length = 1024))			       	
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(MyDense(filters=params[2], Bias=True, Method = 'Float', WidthIn=8, WidthOut=8, Str_Len=16))                           
        model.add(Activation('relu'))
        model.add(MyDense(filters=params[3], Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=256))                           
        model.add(Activation('relu'))
        model.add(MyDense(filters=num_classes, Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=1024))                           
        model.add(Activation('softmax'))


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
        # print("Weigths are loaded from disk")
        
        if is_attack:
            adv_data = fgsm_attack(model, data.x_test, data.y_test, epsilon=epsilon )
            test_loss, test_acc = model.evaluate(adv_data.numpy(), data.y_test)
            test_pred = model.predict(adv_data.numpy())
            print("attack activated?::::::::::::::::::::: YEP")
        else:
            test_loss, test_acc = model.evaluate(data.x_test, data.y_test)
            test_pred = model.predict(data.x_test)
            print("attack activated?::::::::::::::::::::: NEI")
            

        print("we are in SC:", is_SC)
        print(f'Test accuracy: {test_acc}') #, Test loss: {test_loss}')
        # print("----------------------------------------------------------------------------------------")

        # Conf_matrix(data.y_test, test_pred)


    #----for both train and inference	
    # print('batch_size',batch_size,'epochs',epochs)	  
    #score = model.evaluate(x_test, y_test, verbose=0)
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
    #print('Number of Errors:', test_samples - (score[1]*test_samples),'out of ',test_samples)


def attack_main():
    return 

def fgsm_attack(model, x_in, y_in, epsilon):
    
    x = tf.convert_to_tensor(x_in)
    y = tf.convert_to_tensor(y_in)
    
    # Create a TensorFlow GradientTape to compute gradients
    with tf.GradientTape() as tape:
        tape.watch(x)  # Mark x as a trainable tensor
        predictions = model(x)
        loss = losses.CategoricalCrossentropy()(y, predictions) #SparseCategoricalCrossentropy(from_logits=True)(y, predictions)

    # Calculate gradients of the loss w.r.t the input image
    gradients = tape.gradient(loss, x)

    # Get the sign of the gradients
    perturbations = epsilon * tf.sign(gradients)

    # Add perturbations to the original image
    x_adv = x + perturbations

    # Clip the adversarial image to ensure valid pixel values
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    return x_adv





#load data once
data_set = Load_DATA(train_samples=train_samples, test_samples=test_samples, 
                      validation_samples=validation_samples, img_rows=img_rows, img_cols=img_cols, 
                      num_classes=num_classes)
print("done 1 : data loaded -----------------------")

# #now train the binary Lenet model
# train_main(data_set, train_1_test_2=1, is_SC=False, params=params)
# print("done 3 : nn trained -----------------------")

# #test the stochastic model
# train_main(data_set, train_1_test_2=2, is_SC=True, params=params)
# print("done 4 : SC tested -----------------------")
# # print("----------------------------------------------------------------------------------------")


# #test the stochastic model after fgsm attack
# train_main(data_set, train_1_test_2=2, is_SC=True, params=params, is_attack=True)
# print("done 44 : SC tested -----------------------")
# # print("----------------------------------------------------------------------------------------")



# #test binary model now
# train_main(data_set, train_1_test_2=2, is_SC=False, params=params)
# print("done 5 : nn tested -----------------------")

#test binary model after fgsm s--------------------")
train_main(data_set, train_1_test_2=2, is_SC=False, params=params, is_attack=True)
print("done 5 : nn tested -----------------------")


# # apply attack and save the out in files
# adv_test_data, adv_test_labels = attack_main()
# print("done 6 : attack applied -----------------------")

# data_set.save_after_attack(adv_test_data, adv_test_labels)
# print("done 9 : attack saved in file -----------------------")

# data_set.load_after_attack()
# print("done 10 : attacked data loaded from file -----------------------")

# #test SC network, after attack being applied
# train_main(data_set, train_1_test_2=2, is_SC=True, params=params)
# print("done 7 : SC tested after attack -----------------------")
# print("----------------------------------------------------------------------------------------")


# #test binary network, after attack being applied
# train_main(data_set, train_1_test_2=2, is_SC=False, params=params)
# print("done 8 : nn tested after attack -----------------------")
