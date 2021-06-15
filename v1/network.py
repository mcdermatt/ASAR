import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

"""main document containing networks for movement simulation"""


def Net1(**kwargs):

	"""network for dead reckoning given two sequential lidar scans

	inputs: lidar scans
	outputs: estimated translation (x and y) and rotation"""

	inputs = keras.Input(shape=(100))

	X = keras.layers.Dense(units = 32, activation = 'relu')(inputs) 
	X = keras.layers.BatchNormalization()(X)
	X = keras.layers.Dense(units = 32, activation = 'relu')(X) 
	X = keras.layers.BatchNormalization()(X)
	
	outputs = keras.layers.Dense(units = 3, activation = 'tanh')(X)

	#scale output layer to up to 10 units translation, full rotation
	outputs = outputs*tf.constant([10., 10., np.pi.])

	model = tf.keras.Model(inputs,outputs)

	return model
