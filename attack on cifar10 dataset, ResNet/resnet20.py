import keras
from keras.layers   import Input
from keras 		    import layers
from keras.layers   import Dense, Conv2D, BatchNormalization, Activation, Input, Flatten, Dropout, Lambda
from keras.layers   import MaxPooling2D, AveragePooling2D, GaussianNoise, BatchNormalization
from MyConv2D_layer import MyConv2D, MyActivation
from Stochastic     import Quantizer
from keras 		    import backend as K
from Regularizers   import Er, l2, l1
from keras.models import Model
import tensorflow as tf
class ResNet20:
    #===================================================================================
    def __init__(self, input_shape, depth, num_classes=10, pre_softmax=False):

        # Start model definition.
        num_filters = 16
        #-------------------------------------------------------------
        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs, convConfig = ['Float', 8, 8, [1,4], 128]) #9*3*32*32*16 = 442,368          #128
        #================================================================= Basic Block 0
        strides = 1
        #-------------------------------------------------------------
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*16*32*32*16 = 2,359,296
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,3], 32]) #9*16*32*32*16 = 2,359,296
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #-------------------------------------------------------------
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*16*32*32*16= 2,359,296
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 32]) #9*16*32*32*16= 2,359,296
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #-------------------------------------------------------------
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 32]) #9*16*32*32*16 = 2,359,296
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*16*32*32*16 = 2,359,296
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #================================================================= Basic Block 1
        num_filters *= 2 
        strides = 2  # downsample
        #-------------------------------------------------------------
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*16*16*16*32 = 1,179,648
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*32*16*16*32 = 2,359,296
        #--------------------------- first layer but not first stack, residual shortcut connection
        x = self.resnet_layer(inputs=x, num_filters=num_filters,kernel_size=1, strides=strides, padding = 'valid', convConfig = ['Float', 8, 8, [1,4], 128],
                        activation=None, batch_normalization=False);																	   #1*16*16*16*32 = 131,072
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #-------------------------------------------------------------  
        strides = 1
        #-------------------------------------------------------------
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*32*16*16*32 = 2,359,296
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*32*16*16*32 = 2,359,296
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #-------------------------------------------------------------  
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*32*16*16*32 = 2,359,296
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*32*16*16*32 = 2,359,296
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #================================================================= Basic Block 2   
        num_filters *= 2 
        strides = 2  # downsample
        #-------------------------------------------------------------
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*32*8*8*64 = 1,179,648
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*64*8*8*64 = 2,359,296
        #--------------------------- first layer but not first stack, residual shortcut connection
        x = self.resnet_layer(inputs=x, num_filters=num_filters,kernel_size=1, strides=strides, padding = 'valid', convConfig = ['Float', 8, 8, [1,4], 128],
                        activation=None, batch_normalization=False);																	   #1*32*8*8*64 = 131,072
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #-------------------------------------------------------------  
        strides = 1
        #-------------------------------------------------------------
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*64*8*8*64 = 2,359,296
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*64*8*8*64 = 2,359,296
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #-------------------------------------------------------------  
        y = self.resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*64*8*8*64 = 2,359,296
        y = self.resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same', convConfig = ['Float', 8, 8, [1,4], 128]) #9*64*8*8*64 = 2,359,296
        #---------------------------
        x = keras.layers.add([x, y])
        #---------------------------
        x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
        #------------------------------------------------------------- Add classifier on top
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        #-------------------------------------------------------------
        if pre_softmax:
            outputs = Dense(num_classes, kernel_initializer='he_normal')(y)
        else:
            outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
        #------------------------------------------------------------- Instantiate model.
        self.model = Model(inputs=inputs, outputs=outputs)
        #-------------------------------------------------------------
        # return self.model
    #===================================================================================
    def predict(self, data):
            out = self.model(data)
            return out
    #===================================================================================
    def resnet_layer(self, inputs,
                    num_filters = 16,
                    kernel_size = 3,
                    strides = 1,
                    padding = 'same',
                    activation='relu',
                    batch_normalization = True,
                    bias = False,
                    convConfig = ['Stoch', 8, 8, [1,4], 128],
                    activationConfig = [False, 8, 8]):
        #--------------------------------------- Create a Customized Convolution Layer
        MethodConv, WidthInConv, WidthOutConv, SNG_numConv, Str_LenConv = convConfig;
        MyConv = MyConv2D(filters = num_filters, kernel = (kernel_size, kernel_size), Strides = strides, Padding = padding, Bias = bias, W_regularizer = l2(1e-4),
                        Method = MethodConv, WidthIn = WidthInConv, WidthOut = WidthOutConv, SNG_num = SNG_numConv, Str_Len = Str_LenConv);
        #---------------------------------------  Create a Customized Batch Normalization Layer
        #MyBatch = MyConv2D(Type = activation, Stoch = StochAct, WidthIn = WidthInAct, WidthOut = WidthOutAct);
        #--------------------------------------- Create a Customized Activation Layer
        StochAct, WidthInAct, WidthOutAct = activationConfig;
        MyAct = MyActivation(activation, StochAct, WidthInAct, WidthOutAct);
        
        #--------------------------------------- Evaluate the Convolution Layer
        x = MyConv(inputs)
        #--------------------------------------- Add a Batch-Normalization unit
        if batch_normalization:
            x = BatchNormalization()(x)
        #--------------------------------------- Add an Activation unit
        if activation is not None:
            x = MyAct(x)
            #x = Activation(activation)(x)
        #---------------------------------------
        return x
    #===================================================================================