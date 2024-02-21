#Need this for reproducibility of the results (stop shuffling keras)
# from numpy.random import seed, seed(1) #for reproducibility
# from tensorflow import set_random_seed, set_random_seed(2) #for reproducibility
# PYTHONHASHSEED = 0 #for reproducibility
from __future__ import print_function
# import keras
# import h5py
from keras import losses
# from keras.optimizers 			import Adam, SGD
# from keras.callbacks 			import ModelCheckpoint, LearningRateScheduler
# from keras.callbacks 			import ReduceLROnPlateau
from keras.preprocessing.image  import ImageDataGenerator
# from keras.regularizers 		import l2
from keras 						import backend as K
# from keras.utils 		        import multi_gpu_model
import numpy as np
import os
import keras
# from keras.models   import  Sequential

from Attacker			import MainAttack
from PrepareData  		import DataPreparation, AfterAdvDataPreparation
from resnet20     		import ResNet20
from LearningScheduler  import lr_schedule
import tensorflow as tf


###-----------------------------------INITIATIVES---------------------------------------------
DataSet = 'CIFAR10'
#-------------------------
# Computed depth from supplied model parameter n
n = 3
depth = n * 6 + 2
model_type = 'ResNet%d' %(depth)
#-------------------------
epochs 			= 100 #200
batch_size 		= 32
train_samples	= 60000 #60000
test_samples	= 1000 #10000
num_classes 	= 10
#-------------------------
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = False#True
data_augmentation 	= True

#--------------------------------------------------------------------------------
# Data Pre-Processing
(x_train, y_train), (x_test, y_test), input_shape = DataPreparation(DataSet, train_samples, test_samples, num_classes, subtract_pixel_mean)
#--------------------------------------------------------------------------------




def	Main_Train_Test(Train_Phase=True):

    config = tf.compat.v1.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True							##### NOT SURE

    sess = tf.compat.v1.Session(config=config)
    # K.tensorflow_backend._get_available_gpus()

    # def get_available_gpus():
    # 	"""Returns a list of available GPU devices."""
    # 	gpus = tf.config.list_physical_devices('GPU')
    # 	return [gpu.name for gpu in gpus]

    # get a list of available GPU devices
    available_gpus = tf.config.list_physical_devices('GPU')
    # print("Available GPUs:", available_gpus)

    #--------------------------------------------------------------------------------
    # Call a model and compile the model
    model = ResNet20(input_shape=input_shape, depth= depth).model

    # This assumes that your machine has at least 2 available GPUs.
    # parallel_model = multi_gpu_model(model, gpus=1)
    model.compile(loss=losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule(0)), metrics=['accuracy'])
    # model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=lr_schedule(0)), metrics=['accuracy'])

    # model.summary()
    print(model_type)

    #--------------------------------------------------------------------------------
    # Train or Test trained model
    if Train_Phase:	#------- Trian Phase -------
        #-----------------------------------
        # Prepare model-model saving-directory.
        save_dir = os.path.join(os.getcwd(), 'saved_models_1')
        model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        #-----------------------------------
        # Prepare callbacks for model saving and for learning rate adjustment.
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=False) 
        #-----------------------------------
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
        #-----------------------------------
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.81), cooldown=0, patience=5, min_lr=0.5e-6)
        #-----------------------------------
        callbacks = [checkpoint, lr_reducer, lr_scheduler]
        #-----------------------------------
        if not data_augmentation:
            #------------------------------
            print('Not using data augmentation.')
            #------------------------------
            # Fit the model on the batches.
            model.fit(x_train, y_train,	validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs, 
                shuffle=True, verbose=1, callbacks=callbacks)
            #------------------------------
            model_name = 'cifar10_%s_model.h5' % model_type
            model.save_weights(model_name)
            #------------------------------
        else:
            #------------------------------
            print('Using real-time data augmentation.')
            #------------------------------
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # epsilon for ZCA whitening
                zca_epsilon=1e-06,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # set range for random shear
                shear_range=0.,
                # set range for random zoom
                zoom_range=0.,
                # set range for random channel shifts
                channel_shift_range=0.,
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                # value used for fill_mode = "constant"
                cval=0.,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False,
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)
            #------------------------------
            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)
            #------------------------------
            # Fit the model on the batches generated by datagen.flow().
            # model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
            # 					validation_data=(x_test, y_test),
            # 					epochs=epochs, steps_per_epoch = int(len(x_train)/ batch_size), workers=4,callbacks=callbacks, verbose=1)
            model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
                                validation_data=(x_test, y_test),
                                epochs=epochs, steps_per_epoch = int(len(x_train)/ batch_size), workers=4,callbacks=callbacks, verbose=1)
            #------------------------------
            model_name = 'cifar10_%s_model.h5' % model_type
            model.save_weights(model_name)
            #------------------------------	
    else:			#------- Test Phase --------
        model_name = 'cifar10_%s_model.h5' % model_type
        model.load_weights(model_name)
        print("Weigths are loaded from disk")

    #--------------------------------------------------------------------------------
    # Score trained model.	
    print('batch_size',batch_size,'epochs',epochs)	  
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # print('Number of Errors:', test_samples - (score[1]*test_samples),'out of ',test_samples)

    return



## --------------------  Train ResNet ---------------------------------- ##
Main_Train_Test(Train_Phase=True)


# ## -------------------- Test ResNet With No Attack --------------------- ##
# Main_Train_Test(Train_Phase=False)



# ## --------------------  Attack applies -------------------------------- ##
# MainAttack(model_type, x_test, y_test, input_shape, test_samples)



# # ## ---------------------- Test ResNet With l2 Attack ------------------- ##
# (x_test, y_test) = AfterAdvDataPreparation(y_test)
# Main_Train_Test(Train_Phase=False)