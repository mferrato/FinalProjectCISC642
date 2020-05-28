from keras.layers import Input
from keras.models inport Model
from keras.layers import Activation
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Dense
from bilinear_interpolation import *
import numpy as np


def get_initial_weights(size_of_output):
	b = np.zeros((2, 3), dtype='float32')
	b[0,0] = 1
	b[1,1] = 1
	w = np.zeros((size_of_output, 6), dtype='float32')
	weights = [w, b.flatten()]
	return weights


def STN(input_shape=(60, 60, 1), sampling_size=(30, 30), num_classes=10):
	image = Input(shape=input_shape)
	locnet = MaxPool2D(pool_size=(2, 2))(image)
	locnet = Conv2D(20, (5, 5))(locnet)
	locnet = MaxPool2D(pool_size=(2, 2))(locnet)
	locnet = Conv2D(20, (5, 5))(locnet)
    	locnet = Flatten()(locnet)
    	locnet = Dense(50)(locnet)
    	locnet = Activation('relu')(locnet)
    	weights = get_initial_weights(50)
    	locnet = Dense(6, weights=weights)(locnet)
    	x = BilinearInterpolation(sampling_size)([image, locnet])
    	x = Conv2D(32, (3, 3), padding='same')(x)
    	x = Activation('relu')(x)
    	x = MaxPool2D(pool_size=(2, 2))(x)
    	x = Conv2D(32, (3, 3))(x)
    	x = Activation('relu')(x)
    	x = MaxPool2D(pool_size=(2, 2))(x)
    	x = Flatten()(x)
    	x = Dense(256)(x)
    	x = Activation('relu')(x)
    	x = Dense(num_classes)(x)
    	x = Activation('softmax')(x)
    
	return Model(inputs=image, outputs=x)
