from __future__ import print_function
import tensorflow as tf
from keras import losses
#import h5py
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, Conv2D, MaxPooling2D, Activation, GaussianNoise, BatchNormalization
from keras import backend as K
from MyConv2D_layer import MyConv2D
from MyDense_layer import MyDense


batch_size = 64
num_classes = 10
epochs = 40
train_samples=60000
test_samples=10000
train_1_test_2 = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# select a subset of the dataset for the train and test
x_train = x_train[0:train_samples];y_train = y_train[0:train_samples];
#x_test = x_test[0:5000];y_test = y_test[0:5000];
x_test = x_test[0:test_samples];y_test = y_test[0:test_samples];


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

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

model = Sequential()

model.add(MyConv2D(filters=20, kernel=(5, 5), Method = 'Stoch', Width = 8, Sobol_num1 = 1, Sobol_num2 = 4, Stream_Length = 16, input_shape=input_shape))		 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(MyConv2D(filters=20, kernel=(5, 5), Method = 'Fixed', Width = 8, Sobol_num1 = 2, Sobol_num2 = 4, Stream_Length = 64))			        #used to be Fixed	
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(MyDense(filters=800, Bias=True, Method = 'Fixed', WidthIn=8, WidthOut=8, Str_Len=256))                                                #used to be Fixed
model.add(Activation('relu'))
model.add(MyDense(filters=10, Bias=True, Method = 'Fixed',  WidthIn=8, WidthOut=8, Str_Len=256))                                                #used to be Fixed
model.add(Activation('relu'))
model.add(MyDense(filters=10, Bias=True, Method = 'Float',  WidthIn=8, WidthOut=8, Str_Len=256))                                                #used to be Fixed
model.add(Activation('softmax'))

##model.add(Conv2D(14, (5, 5), activation='relu', use_bias=True, bias_initializer='RandomNormal'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(20, (5, 5), activation='relu', use_bias=True, bias_initializer='RandomNormal'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(800, input_dim=320, activation='relu', use_bias=True, bias_initializer='RandomNormal'))
#model.add(Dense(10, input_dim=800, activation='relu', use_bias=True, bias_initializer='RandomNormal'))
#model.add(Dense(num_classes, activation='softmax', use_bias=True, bias_initializer='RandomNormal'))

#print model summary
#model.summary()


model.compile(loss=losses.CategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

if train_1_test_2==1: ##--only for train
    #----Saving the best training weights 'my_LeNet5_best.h5'
	filepath = 'my_LeNet5_best.h5'; #Save the best epoch
    #----options for monitor: 'val_loss', 'val_accuracy', ...(it seems that 'val_accuracy' works better for SC)
	model_saver_best = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, 
                                                        save_best_only=True, mode='auto', save_weights_only=True, save_freq='epoch') 
                                                        #options=tf.train.CheckpointOptions()) ,   save_freq=5*batch_size   , period=1
	model.fit(x_train, y_train,	batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test),callbacks=[model_saver_best])
	
    
elif train_1_test_2==2:	  
	#----only for inference
	model.load_weights('my_LeNet5_best.h5')
	print("Weigths are loaded from disk")



model.summary()


#----for both train and inference	
print('batch_size',batch_size,'epochs',epochs)	  
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Number of Errors:', test_samples - (score[1]*test_samples),'out of ',test_samples)



