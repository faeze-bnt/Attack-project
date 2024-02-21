import keras
from keras.layers   import Input
from keras 		    import layers
from keras.layers   import Dense, Conv2D, BatchNormalization, Activation, Input, Flatten, Dropout, Lambda
from keras.layers   import MaxPooling2D, AveragePooling2D, GaussianNoise, BatchNormalization
from MyConv2D_layer import MyConv2D, MyActivation
from Stochastic     import Quantizer
from keras 		    import backend as K
from Regularizers   import Er, l2, l1
from keras.models   import Model
import tensorflow   as tf
#===================================================================================
def resnet(input_shape, depth, num_classes=10):

    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
	#-------------------------------------------------------------
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
	#------------------------------------------------------------- Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
			#-------------------------------------------------------------
            y = resnet_layer(inputs=x, num_filters = num_filters, strides=strides, padding = 'same')
            y = resnet_layer(inputs=y, num_filters = num_filters, activation=None, padding = 'same')
			#------------------------------------------------------------- first layer but not first stack
            if stack > 0 and res_block == 0: # linear projection residual shortcut connection to match changed dims
                x = resnet_layer(inputs=x, num_filters=num_filters,kernel_size=1, strides=strides, padding = 'valid',
                                 activation=None, batch_normalization=False);
			#-------------------------------------------------------------
            x = keras.layers.add([x, y])
			#-------------------------------------------------------------
            x = MyActivation('relu', False, WidthIn = 8, WidthOut = 8)(x)
			#-------------------------------------------------------------
        num_filters *= 2
	#------------------------------------------------------------- Add classifier on top
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
	#-------------------------------------------------------------
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    #------------------------------------------------------------- Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
	#-------------------------------------------------------------
    return model
#===================================================================================
def resnet_layer(inputs,
                 num_filters = 16,
                 kernel_size = 3,
                 strides = 1,
                 padding = 'same',
                 activation='relu',
                 batch_normalization = True,
                 bias = False,
                 convConfig = ['Fixed', 8, 8, [1,4], 32],
                 activationConfig = [False, 8, 8]):
    #--------------------------------------- Create a Customized Convolution Layer
    MethodConv, WidthInConv, WidthOutConv, SNG_numConv, Str_LenConv = convConfig;
    MyConv = MyConv2D(filters = num_filters, kernel = (kernel_size, kernel_size), Strides = strides, Padding = padding, Bias = bias, W_regularizer = l2(1e-4),
                      Method = MethodConv, WidthIn = WidthInConv, WidthOut = WidthOutConv, SNG_num = SNG_numConv, Str_Len = Str_LenConv);
    #---------------------------------------  Create a Customized Batch Normalization Layer
    #MyBatch = MyConv2D(Type = activation, Fixed = FixedAct, WidthIn = WidthInAct, WidthOut = WidthOutAct);
    #--------------------------------------- Create a Customized Activation Layer
    FixedAct, WidthInAct, WidthOutAct = activationConfig;
    MyAct = MyActivation(activation, FixedAct, WidthInAct, WidthOutAct);
    
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