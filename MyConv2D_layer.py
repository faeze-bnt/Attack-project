#Source: https://medium.com/@konpat/2d-convolution-layer-without-conv2d-in-keras-8da47cd73e4e
import numpy as np
#np.random.seed(1)
from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras import backend as K
from keras import initializers
import tensorflow as tf
from Stochastic   import Stochastic_SNG_A, Stochastic_SNG_B, Quantizer
from keras import regularizers
from tensorflow.python.framework import tensor_shape

class MyConv2D(layers.Layer):
    '''
    Implemeting a Conv2D with strides=1, and 'valid' padding
    '''
    
    def __init__(self, filters, kernel, Method, Width, Sobol_num1, Sobol_num2, Stream_Length, **kwargs):
        self.filters = filters
        self.k_h, self.k_w = kernel
        self.Method = Method
        self.Width = Width
        self.Sobol_num1 = Sobol_num1
        self.Sobol_num2 = Sobol_num2
        self.Stream_Length = Stream_Length
        self.kernel_initializer = initializers.RandomNormal(stddev=0.01) #'glorot_uniform'#'RandomNormal'
        super(MyConv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.h, self.w, self.c = input_shape
		
        # Expected output size
        self.out_h = self.h - self.k_h + 1
        self.out_w = self.w - self.k_w + 1
        self.output_dim = self.out_h, self.out_w, self.filters
		
        # Create a trainable weight variable for this layer.
        self.kernel_size = self.k_h * self.k_w * self.c
        self.kernels = self.add_weight(name='kernel',
                                       shape=[self.k_h, self.k_w, self.c, self.filters],
                                       initializer=self.kernel_initializer,
                                       trainable=True)
        self.bias = self.add_weight(name='bias',
                                shape=(self.output_dim),
                                initializer=initializers.RandomNormal(stddev=0.01),#'RandomNormal',
                                trainable=True)
		
        super(MyConv2D, self).build(input_shape)

    def call(self, x):
        #------------------------------------------------------ Configuration parameters
        #Method = 2            # 1: Float, 2: Binary-Fixed, 3: Stochastic
        #Width  = 8         	   # precision
        #Sobol_num1 = 1         # Sobol Sequence Number 1
        #Sobol_num2 = 2     	   # Sobol Sequence Number 2
        #Stream_Length = 8 # Stream Length, 2**(2*Width)
		#-------------------------------------------------------------------------------
        # flatten kernels [k_h, k_w, c_in, c_out] -> [k_h * k_w * c_in, c_out]
        
        #kernel = K.reshape(self.kernels, [self.kernel_size, self.filters])
        
        #-------------------------------------------------------------------------------
        if self.Method == 'Float':     # Method 1 - convolution: using original keras bulit-in function
            output = K.conv2d(x, self.kernels, padding='valid')       # Do covolution using keras bulit-in function with floating-point inputs
        elif self.Method == 'Fixed':                                             
            x = Quantizer(x, self.Width) 							      # Quantize Input
            kernels = Quantizer(self.kernels, self.Width)
            #self.kernels = Quantizer(self.kernels, self.Width)  	      # Quantize Kernels
            output1 = K.conv2d(x, kernels, padding='valid')     	  # Do covolution using keras bulit-in function with Quantizer inputs
            output = Quantizer(output1, self.Width)
            #output = output1
        elif self.Method == 'Stoch':
            x = Quantizer(x, self.Width)
            kernels = Quantizer(self.kernels, self.Width)        
        	#-------------- Stochastic bit-stream generators		
            A_stream = Stochastic_SNG_A(x, self.Stream_Length, self.Sobol_num1, 0)
            B_stream = Stochastic_SNG_B(kernels, self.Stream_Length, self.Sobol_num2)
            #-------------- Stochastic convolution
            output1 = K.conv2d(A_stream, B_stream, padding='valid')
            #-------------------------------------------------------------------------------
            output = Quantizer(output1/self.Stream_Length, self.Width)
        output = K.relu(output + self.bias)
        return output
        #return Quantizer(output, self.Width);

    def compute_output_shape(self, input_shape):
        return (None, self.out_h, self.out_w, self.filters)
        
