#Source: https://medium.com/@konpat/2d-convolution-layer-without-conv2d-in-keras-8da47cd73e4e
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import  BatchNormalization, Activation
from keras 		  import layers
from keras 		  import backend as K
from keras.regularizers import l2
from Stochastic   import Stochastic_SNG_A, Stochastic_SNG_B, Quantizer, QuantizerWeight

#===================================================================================
class MyConv2D(layers.Layer):    
    def __init__(self, filters, kernel, Strides = 1, Padding = 'valid', Bias = False, W_regularizer = None, B_regularizer = None, Method = 2, WidthIn = 8, WidthOut = 8, SNG_num = [1,2], Str_Len = 8, **kwargs):
        self.filters  = filters
        self.k_h, self.k_w = kernel
        self.Strides  = Strides
        self.Padding  = Padding
        self.Bias  = Bias
        self.W_regularizer = W_regularizer
        self.B_regularizer = B_regularizer
        self.Method   = Method
        self.WidthIn  = WidthIn
        self.WidthOut = WidthOut
        self.SNG_num  = SNG_num
        self.Str_Len  = Str_Len
        self.kernel_initializer = 'RandomNormal'
        super(MyConv2D, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel": [self.k_h, self.k_w],
            "Strides": self.Strides,
            "Padding": self.Padding,
            "Bias": self.Bias,
            "W_regularizer": self.W_regularizer,
            "B_regularizer": self.B_regularizer,
            "Method": self.Method,
            "WidthIn": self.WidthIn,
            "WidthOut": self.WidthOut,
            "SNG_num": self.SNG_num,
            "Str_Len": self.Str_Len,
        })
        return config
    
    def build(self, input_shape):

        _, self.h, self.w, self.c = input_shape

        #----------------
        if self.Padding == 'valid':
            self.p = 0
        else:
            self.p = 1
        #----------------
        # Expected output size
        self.out_h = int((self.h - self.k_h + 2*self.p)/self.Strides) + 1
        self.out_w = int((self.w - self.k_w + 2*self.p)/self.Strides) + 1
        self.output_dim = self.out_h, self.out_w, self.filters
		
        # Create a trainable weight variable for this layer.
        self.kernels = self.add_weight(name='kernel',
                                       shape=[self.k_h, self.k_w, self.c, self.filters],
                                       initializer=self.kernel_initializer,
                                       regularizer=self.W_regularizer,
                                       trainable=True)
        if self.Bias:
           self.bias = self.add_weight(name='bias',
									   shape=(self.output_dim),
									   initializer='RandomNormal',
									   regularizer=self.B_regularizer,
									   trainable=True)
		
        super(MyConv2D, self).build(input_shape)

    def call(self, x):
        #------------------------------------------------------ Configuration mode
        #Method = 1: Float, 2: Binary-Fixed, 3: Stochastic
        #-------------------------------------------------------------------------------
        if self.Method == 'Float':     # Method 1 - convolution: using original keras bulit-in function
            output = K.conv2d(x, self.kernels, strides=(self.Strides,self.Strides), padding=self.Padding)    # Do covolution using keras bulit-in function with floating-point inputs
        elif self.Method == 'Fixed':                                             
            x = Quantizer(x, self.WidthIn) 							   # Quantize Input
            kernels = Quantizer(self.kernels, self.WidthIn, 'round', clip = True)  	       # Quantize Kernels
            output1 = K.conv2d(x, kernels, strides=(self.Strides,self.Strides), padding=self.Padding)   # Do covolution using keras bulit-in function with Quantizer inputs
            output = Quantizer(output1, self.WidthOut, 'floor', clip = False)
        elif self.Method == 'Fixed2':                                             
            x = Quantizer(x, self.WidthIn) 							   # Quantize Input
            kernels = QuantizerWeight(self.kernels, self.WidthIn, 'round', clip = True)  	       # Quantize Kernels
            output1 = K.conv2d(x, kernels, strides=(self.Strides,self.Strides), padding=self.Padding)   # Do covolution using keras bulit-in function with Quantizer inputs
            output = Quantizer(output1, self.WidthOut, 'floor', clip = False)
        elif self.Method == 'Stoch':                                             
            x = Quantizer(x, self.WidthIn) 							   # Quantize Input
            kernels = Quantizer(self.kernels, self.WidthIn, 'round', clip = True)  		   # Quantize Kernels      
			#-------------- Stochastic bit-stream generators			
            A_stream = Stochastic_SNG_A(x, self.Str_Len, self.SNG_num[0])
            B_stream = Stochastic_SNG_B(kernels, self.Str_Len, self.SNG_num[1])
			#-------------- Stochastic convolution
            output = K.conv2d(A_stream, B_stream, (self.Strides,self.Strides), padding=self.Padding, data_format='channels_last')/self.Str_Len
			#-------------- Quantizing the output
            output = Quantizer(output, self.WidthOut, 'floor', clip = False)
            #-------------------------------------------------------------------------------
        if self.Bias:
            if self.Method == 'Float':
                return (output + self.bias)
            else:
                return Quantizer(output + self.bias, self.WidthOut, 'floor', clip = False)
        else:
            return output

    def compute_output_shape(self, input_shape):
        return (None, self.out_h, self.out_w, self.filters)
#===================================================================================
class MyBatchNormalization(layers.Layer):
    def __init__(self, Type = 'relu', Fixed = True, WidthIn = 8, WidthOut = 8, **kwargs):
        self.Type = Type
        self.Fixed = Fixed
        self.WidthIn = WidthIn
        self.WidthOut = WidthOut
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BatchNormalization, self).build(input_shape)

    def call(self, x):
        #--------------------------
        output = Activation('relu')(x)
        if self.Fixed:
            return Quantizer(output, self.WidthOut)
        else:
            return output
    def compute_output_shape(self, input_shape):
        return input_shape
#===================================================================================
class MyActivation(layers.Layer):
    def __init__(self, Type = 'relu', Fixed = True, WidthIn = 8, WidthOut = 8, **kwargs):
        self.Type = Type
        self.Fixed = Fixed
        self.WidthIn = WidthIn
        self.WidthOut = WidthOut
        super(MyActivation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MyActivation, self).build(input_shape)

    def call(self, x):
        #--------------------------
        output = Activation(self.Type)(x)
        if self.Fixed:
            return Quantizer(output, self.WidthOut, 'floor', clip = False)
        else:
            return output
    def compute_output_shape(self, input_shape):
        return input_shape
#===================================================================================