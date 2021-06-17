import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

"""main document containing networks for movement simulation"""


#TODO:
#	input data in 2D format so there can be built in association between neighboring points??
#	modify output to equally weight all three

def Net1(**kwargs):

	"""network for dead reckoning given two sequential lidar scans

	inputs: lidar scans
	outputs: estimated translation (x and y) and rotation"""

	l1 = 256
	l2 = 128
	conv_filters = 128 #8
	inshape = 50

	inputs = keras.Input(shape=(inshape,1))

	#test new conv stuff
	X = keras.layers.BatchNormalization()(inputs)
	X = tf.keras.layers.Conv1D(filters = conv_filters, kernel_size = 5, strides = 1, activation='relu')(X)
	X = keras.layers.BatchNormalization()(X)
	X = tf.keras.layers.Conv1D(filters = conv_filters, kernel_size = 3, strides = 1, activation='relu')(X)
	# X = keras.layers.BatchNormalization()(X)
	# X = tf.keras.layers.Conv1D(filters = conv_filters, kernel_size = 7, strides = 3, activation='relu')(X)

	#standard linear layers
	X = keras.layers.Flatten()(X)
	X = keras.layers.Dense(units = l1, activation = 'relu')(X) 
	X = keras.layers.BatchNormalization()(X)
	X = keras.layers.Dense(units = l2, activation = 'relu')(X) 
	X = keras.layers.BatchNormalization()(X)
	
	outputs = keras.layers.Dense(units = 3, activation = 'tanh')(X)

	#scale output layer to up to 10 units translation, full rotation
	outputs = outputs*tf.constant([10., 10., 0.3]) #[..., dx, dy, rotation]
													 
	model = tf.keras.Model(inputs,outputs)

	return model

def Net2(**kwargs):

	"""testnet

	inputs: lidar scans
	outputs: estimated translation (x and y) and rotation"""

	l1 = 256
	l2 = 128
	conv_filters = 128 #8
	inshape = 100

	inputs = keras.Input(shape=(inshape,1))

	#test new conv stuff
	X = keras.layers.BatchNormalization()(inputs)
	X = tf.keras.layers.Conv1D(filters = conv_filters, kernel_size = 5, strides = 1, activation='relu')(X)
	X = keras.layers.BatchNormalization()(X)
	X = tf.keras.layers.Conv1D(filters = conv_filters, kernel_size = 3, strides = 1, activation='relu')(X)
	# X = keras.layers.BatchNormalization()(X)
	# X = tf.keras.layers.Conv1D(filters = conv_filters, kernel_size = 7, strides = 3, activation='relu')(X)

	#standard linear layers
	X = keras.layers.Flatten()(X)
	X = keras.layers.Dense(units = l1, activation = 'relu')(X) 
	X = keras.layers.BatchNormalization()(X)
	X = keras.layers.Dense(units = l2, activation = 'relu')(X) 
	X = keras.layers.BatchNormalization()(X)
	
	outputs = keras.layers.Dense(units = 3, activation = 'tanh')(X)

	#scale output layer to up to 10 units translation, full rotation
	outputs = outputs*tf.constant([10., 10., 0.3]) #[..., dx, dy, rotation]
													 
	model = tf.keras.Model(inputs,outputs)

	return model
