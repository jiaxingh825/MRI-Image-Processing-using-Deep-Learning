import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.optimizers import Adam
from keras.metrics import RootMeanSquaredError
import matplotlib.pyplot as plt

#batch size, iteration number, learning rate
bs = 16
epoch_num = 400
lr = 0.0002

# 5 layer U-net CNN
# Input: N - number of points in each channel
def get_model(N):
    input = tf.keras.Input(shape = (2,N,2))
    x = tfl.Conv2D(128, (11,11), strides=1, padding='same')(input)#kernel_regularizer=l2(0.01)xx
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    #x = tfl.Dropout(0.2)(x)
    x = tfl.Conv2D(64,(9,9),strides=1, padding = "same" )(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    #x = tfl.Dropout(0.2)(x)
    x = tfl.Conv2D(32,(5,5),strides=1, padding = "same" )(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    #x = tfl.Dropout(0.2)(x)
    x = tfl.Conv2D(32,(1,1),strides=1, padding = "same" )(x)
    x = tfl.BatchNormalization()(x)
    x = tfl.ReLU()(x)
    #x = tfl.Dropout(0.2)(x)
    x = tfl.Conv2D(16,(7,7),strides=1, padding = "same" )(x)
    #x = tfl.Flatten()(x)
    #x = tfl.Dense((16*512))(x)
    #x = tfl.Reshape((32,256))(x)
    model = tf.keras.Model(input, outputs=x)
    return model

