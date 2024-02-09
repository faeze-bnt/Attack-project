#Source: https://medium.com/@konpat/2d-convolution-layer-without-conv2d-in-keras-8da47cd73e4e
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import  BatchNormalization, Activation
from keras 		  import layers
from keras 		  import backend as K
from keras.regularizers import l2
from Stochastic   import Stochastic_SNG_A, Stochastic_SNG_B, Quantizer
import tensorflow as tf

#===================================================================================
class MyDense(layers.Layer):

  def __init__(self, filters, Bias = False, Weight_initializer = 'RandomNormal', 
                Bias_initializer = 'RandomNormal', W_regularizer = None, B_regularizer = None, 
                Method = 2, WidthIn = 8, WidthOut = 8, SNG_num = [1,2], Str_Len = 8, **kwargs):
    self.output_dim = filters
    self.Bias = Bias
    self.Weight_initializer = Weight_initializer
    self.Bias_initializer = Bias_initializer
    self.W_regularizer = W_regularizer
    self.B_regularizer = B_regularizer
    self.Method   = Method
    self.WidthIn  = WidthIn
    self.WidthOut = WidthOut
    self.SNG_num  = SNG_num
    self.Str_Len  = Str_Len
    super(MyDense, self).__init__(**kwargs)

  def build(self, input_shape):

    _, self.l = input_shape

	# Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
								   shape=[input_shape[-1], self.output_dim],
								   initializer=self.Weight_initializer,
								   regularizer=self.W_regularizer,
								   trainable=True);
    if self.Bias:
       self.bias = self.add_weight(name='bias',
								   shape=[self.output_dim,],
								   initializer=self.Bias_initializer,
								   regularizer=self.B_regularizer,
								   trainable=True);
	
    super(MyDense, self).build(input_shape)

  def call(self, x):
	#------------------------------------------------------ Configuration mode
	#Method = 1: Float, 2: Binary-Fixed, 3: Stochastic
	#-------------------------------------------------------------------------------
    if self.Method == 'Float':     # Method 1 - convolution: using original keras bulit-in function
        # output = tf.experimental.numpy.matmul(x, self.kernel)
        output = tf.matmul(x, self.kernel)    # Do multiplication using keras bulit-in function with floating-point inputs and output
    elif self.Method == 'Fixed':                                             
        x = Quantizer(x, self.WidthIn)                         							                # Quantize Input
        kernel = Quantizer(self.kernel, self.WidthIn, type='round')   	                    # Quantize Kernel 
        output = tf.matmul(x, kernel)                                                   # Do multiplication using keras bulit-in function with Quantizer inputs and output
        output = Quantizer(output, self.WidthOut, clip = False)
    elif self.Method == 'Stoch':                                             
        x = Quantizer(x, self.WidthIn) 							                                        # Quantize Input
        kernel = Quantizer(self.kernel, self.WidthIn, type='round')  	                      # Quantize Kernel
        #-------------- Stochastic bit-stream generators	
        A_stream = Stochastic_SNG_A(x, self.Str_Len, self.SNG_num[0], self.l)
        B_stream = Stochastic_SNG_B(kernel, self.Str_Len, self.SNG_num[1])
        #-------------- Stochastic convolution
        output = tf.matmul(A_stream, B_stream)/self.Str_Len
        #-------------- Quantizing the output
        output = Quantizer(output, self.WidthOut, clip = False)
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
    #return (None, self.output_dim)
#===================================================================================