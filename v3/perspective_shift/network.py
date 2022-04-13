import tensorflow as tf
import tensorflow.keras as keras

def Net(**kwargs):

    ''' Test netowrk for identifying perspective shift on toy dataset 
    '''


    inputs = keras.Input(shape=(50, 3)) 

    # X = tf.keras.layers.Reshape((50, 3, 1))(inputs)

    X = keras.layers.BatchNormalization()(inputs)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)

    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)

    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 256, activation = 'relu')(X)

    X = tf.keras.layers.Reshape((50, 8, 8))(X)
    X = tf.keras.layers.MaxPooling2D(pool_size = (10,2), strides = None, padding = 'same')(X)
    # X = tf.keras.layers.AveragePooling2D(pool_size = (10,10), strides = None, padding = 'same')(X)
    ## X = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None, **kwargs)(X)

    # X = tf.keras.layers.Reshape((25, 4, 4, 4))(X)
    # X = keras.layers.MaxPool3D(pool_size = (25,2,2), padding = 'same')(X)


    # X = keras.layers.AveragePooling1D(pool_size = 2, strides = None, padding = 'same')(X)
    # X = keras.layers.Conv1D(filters = 9, kernel_size = 3, padding = 'same')(X)

    X = keras.layers.Flatten()(X)    

    X = keras.layers.Dense(units = 32, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(units = 32, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    

    output = keras.layers.Dense(units=3, activation = 'tanh')(X)

    #rescale output
    output = output*tf.constant([15., 15., 15.])

    model = tf.keras.Model(inputs,output)

    return model