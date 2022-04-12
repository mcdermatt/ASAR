import tensorflow as tf
import tensorflow.keras as keras

def Net(**kwargs):

    ''' Test netowrk for identifying perspective shift on toy dataset 
    '''


    inputs = keras.Input(shape=(50,3)) 

    X = keras.layers.BatchNormalization()(inputs)
    
    X = keras.layers.Flatten()(X)
    
    X = keras.layers.Dense(units = 16, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(units = 16, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    output = keras.layers.Dense(units=3, activation = 'tanh')(X)

    #rescale output
    output = output*tf.constant([30., 30., 0.3])

    model = tf.keras.Model(inputs,output)

    return model