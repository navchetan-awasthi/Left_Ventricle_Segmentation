# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:23:14 2021

@author: 20181758
"""
import tensorflow
#import miscnn
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv3DTranspose
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Concatenate, Add, concatenate
from tensorflow.keras.layers import ReLU, ELU
from tensorflow.keras.initializers import glorot_normal, Identity
#from tensorflow.keras.contrib.layers import repeat
#from tensorflow.keras.contrib.framework import Arg_Scope
from tensorflow.keras.regularizers import l2

#from miscnn.neural_network.architecture.abstract_architecture import Abstract_Architecture

fc = Dense
conv = Conv2D
deconv = Conv2DTranspose
relu = ELU
maxpool = MaxPooling2D
dropout_layer = Dropout
batchnorm = BatchNormalization
winit = glorot_normal()
#repeat = repeat
#arg_scope = Aarg_Scope
l2_regularizer = l2

import tensorflow as tf
#from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, concatenate
def Dilationlayer(inputspre,n_filters,filter_size=1,dilation_rate=1,dropout=0.2):
    input_shape=inputspre.shape
#     lay1 = tf.keras.layers.BatchNormalization()(inputspre)
    lay1 = BatchNormalization(momentum=0.99)(inputspre)
#     lay2 = relu(lay1)
    lay3 = tf.keras.layers.Conv2D(n_filters, (filter_size,1), strides=(1,1),dilation_rate=dilation_rate,activation='relu', input_shape=input_shape[1:],padding = 'same')(lay1)
    lay3 = tf.keras.layers.Conv2D(n_filters, (1,filter_size),strides=(1,1),dilation_rate=dilation_rate,input_shape=input_shape[1:],padding = 'same')(lay3)
    lay5 = tf.keras.layers.Dropout(rate=dropout)(lay3)
#     lay=tf.keras.layers.concatenate([inputspre,lay5])
    lay = concatenate([inputspre, lay5], axis=-1)

    return lay
def downsampleBlock(inputdowns, n_filters, dropout=0.2):
    
#     lay1 = tf.keras.layers.BatchNormalization()(inputdowns)
    lay1 = BatchNormalization(momentum=0.99)(inputdowns)
#     lay2 = relu(lay1)
#     lay3 = tf.keras.layers.Conv2D(n_filters, (1,1), strides=(1,1), padding='same', activation=None, 
#                          dilation_rate=1, use_bias=False, kernel_regularizer=tf.keras.regularizers.L2(0.00004))(lay2) #Do I need to add regularization??
    lay3 = tf.keras.layers.Conv2D(n_filters, (1,1),  strides=(1,1), padding='same', activation='relu', 
                         dilation_rate=1, use_bias=False)(lay1) #Do I need to add regularization??

    lay4=  tf.keras.layers.Dropout(rate=dropout)(lay3)
    lay5=  tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(lay4)
   
    return lay5


def DenseDilatedBlock(inputdense, n_filters,filter_size, dilation_rate):
    lay1=tf.keras.layers.Conv2D(n_filters,[filter_size,1], strides=(1,1),dilation_rate=dilation_rate[0],padding='same',activation='relu')(inputdense)
    lay1=tf.keras.layers.Conv2D(n_filters,[1,filter_size],strides=(1,1),dilation_rate=dilation_rate[0],padding='same',activation='relu')(lay1)
#     lay1i=tf.keras.layers.concatenate([inputdense,lay1])
    lay1i = concatenate([inputdense, lay1], axis=-1)
    
    lay2=tf.keras.layers.Conv2D(n_filters,[filter_size,1],strides=(1,1), dilation_rate=dilation_rate[1],padding='same',activation='relu')(lay1i)
    lay2=tf.keras.layers.Conv2D(n_filters,[1,filter_size],strides=(1,1), dilation_rate=dilation_rate[1],padding='same',activation='relu')(lay2)
#     lay12=tf.keras.layers.concatenate([lay1i,lay2])
    lay12 = concatenate([lay1i, lay2], axis=-1)


    lay3=tf.keras.layers.Conv2D(n_filters,[filter_size,1],strides=(1,1), dilation_rate=dilation_rate[2],padding='same',activation='relu')(lay12)
    lay3=tf.keras.layers.Conv2D(n_filters,[1,filter_size],strides=(1,1), dilation_rate=dilation_rate[2],padding='same',activation='relu')(lay3)
#     lay312=tf.keras.layers.concatenate([lay3,lay12])
    lay312 = concatenate([lay3, lay2], axis=-1)
    lay = concatenate([lay312, lay1], axis=-1)

    
#     lay4=tf.keras.layers.Conv2D(n_filters,[filter_size,1],strides=(1,1), dilation_rate=dilation_rate[3],padding='same',activation='relu')(lay312)
#     lay4=tf.keras.layers.Conv2D(n_filters,[1,filter_size],strides=(1,1), dilation_rate=dilation_rate[3],padding='same',activation='relu')(lay4)
# #     lay=tf.keras.layers.concatenate([lay312,lay4])
#     lay = concatenate([lay312, lay4], axis=-1)

    return lay
def splitBlock(input, n_filters,dropout=0.0, dilation_rate=1):
    x1nd= DepthwiseConv2D(3, strides=(1, 1), dilation_rate=1,depth_multiplier=1,  activation='relu', padding='same', use_bias=False)(input)
    x1nd = BatchNormalization(momentum=0.99)(x1nd)
    x1d=DepthwiseConv2D(3, strides=(1, 1), dilation_rate=dilation_rate,depth_multiplier=1,  activation='relu', padding='same', use_bias=False)(input)
    x1d=BatchNormalization(momentum=0.99)(x1d)
    x1=tf.keras.layers.add([x1nd,x1d])
    x1 = Conv2D(n_filters, 1, strides=1, padding='same', activation='relu', kernel_initializer=winit,
                         dilation_rate=1, use_bias=False, kernel_regularizer=l2_regularizer(0.00004))(x1)
    x1=BatchNormalization(momentum=0.99)(x1)
    x2 = Conv2D(n_filters, (1, 3), strides=1, dilation_rate=1,padding='same', activation='relu')(input)
    x2 = Conv2D(n_filters, (3, 1), strides=1, dilation_rate=1,padding='same', activation='relu')(x2)
    x2 = BatchNormalization(momentum=0.99)(x2)
    x2 = Conv2D(n_filters, (1, 3), strides=1, dilation_rate=dilation_rate,padding='same', activation='relu')(x2)
    x2 = Conv2D(n_filters, (3, 1), strides=1, dilation_rate=dilation_rate,padding='same', activation='relu')(x2)
    x2 = BatchNormalization(momentum=0.99)(x2)
    x=concatenate([x2, x1], axis=-1)
    if input.shape[3] == x.shape[3]:
        x=tf.keras.layers.add([input, x])
    x=BatchNormalization(momentum=0.99)(x)
    return x 
    
def LVNET(inputs,k=16): 
    fc1 = tf.keras.layers.Conv2D(2*k,(3,3), dilation_rate=1,strides=(1,1),padding='same',activation='relu')(inputs)
    
    fc2x= splitBlock(fc1, n_filters=2*k, dropout=0.0, dilation_rate=2)
    fc2d=downsampleBlock(fc2x,2*k,dropout=0.0)
    fc2 = Dilationlayer(fc2d,2*k, filter_size=3, dilation_rate=1,dropout=0.0)  
    fc3 = downsampleBlock(fc2,4*k,dropout=0.0)
    fc3dense = DenseDilatedBlock(fc3,4*k,filter_size=3, dilation_rate=[2,4,8])
    fc3x= splitBlock(fc3dense, n_filters=4*k, dropout=0.0, dilation_rate=3)
    
    fc4x= splitBlock(fc1, n_filters=2*k, dropout=0.0, dilation_rate=2) 
    fc4d=downsampleBlock(fc4x,2*k,dropout=0.0)
    fc4 = Dilationlayer(fc4d,2*k, filter_size=3, dilation_rate=1,dropout=0.0)   
    fc5 = downsampleBlock(fc4,4*k,dropout=0.0)
    fc6 = DenseDilatedBlock(fc5,4*k,filter_size=3, dilation_rate=[2,4,8])
    fc6x= splitBlock(fc6, n_filters=4*k, dropout=0.0, dilation_rate=3)
    
    fc7c=concatenate([fc3x,fc6x],axis=-1)
    fc7 = downsampleBlock(fc7c,8*k, dropout=0.0)
    
    fc8 = Dilationlayer(fc7,8*k,filter_size=3, dilation_rate=2,dropout=0.0)
    fc8x= splitBlock(fc8, n_filters=8*k, dropout=0.0, dilation_rate=4)
    fc10 =splitBlock(fc8, n_filters=8*k, dropout=0.0, dilation_rate=6)
    fc14 = Dilationlayer(fc10,8*k, filter_size=3, dilation_rate=2,dropout=0.0)
    fc9 =  tf.keras.layers.Conv2DTranspose(4*k,(3,3), strides=2,padding='same')(fc14)
    fc9= BatchNormalization(momentum=0.99)(fc9)
    fc11 =  tf.keras.layers.Conv2DTranspose(2*k,(3,3), strides=2,padding='same')(fc9)
    fc11= BatchNormalization(momentum=0.99)(fc11)
    fco =  tf.keras.layers.Conv2DTranspose(1*k,(3,3), strides=2,padding='same')(fc11)
    fco= BatchNormalization(momentum=0.99)(fco)
    fco = tf.keras.layers.Conv2D(8*k,(3,3), dilation_rate=1,strides=(1,1),padding='same',activation='relu')(fco)
    output= tf.keras.layers.Conv2D(1,(1,1),strides=1,padding='same',activation='sigmoid',)(fco)
    FcdDN = tf.keras.Model(inputs =[inputs] , outputs = [output])
    return FcdDN
    