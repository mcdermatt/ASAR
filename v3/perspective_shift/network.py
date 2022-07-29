import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

#TODO: figure out whats going on with conv1D not adjusting output shape

def PCRnet(**kwargs):

    '''benchmark of https://github.com/vinits5/pcrnet '''

    insize = 512 #50

    inputs = keras.Input(shape=(insize, 3)) 

    X = tf.expand_dims(inputs, -1)

    X = tf.keras.layers.Conv2D(64, [1,3], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(128, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.reshape(X, [-1,insize,128])

    X = keras.layers.MaxPool1D(pool_size = int(insize/2))(X)
    # X = keras.layers.MaxPool1D(pool_size = int(insize))(X)    #test

    X = keras.layers.Flatten()(X)

    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    output = keras.layers.Dense(units=6, activation = 'tanh')(X) #translation + rotation

    #scale for toilet benchmark
    output = output*tf.constant([3., 3., 3., 3., 3., 3.])

    model = tf.keras.Model(inputs,output)

    return model

def permNet(**kwargs):
    ''' Test nework for attempting permutation invariant input'''

    #https://stackoverflow.com/questions/43151775/how-to-have-parallel-convolutional-layers-in-keras

    #TODO: figure out how to share weights between layers

    inputs = keras.Input(shape = (50,3))

    #define feedfoward layers that will share weights
    ff1 = keras.layers.Dense(32, activation = "relu")
    ff2 = keras.layers.Dense(64, activation = "relu")
    # ff3 = keras.layers.Dense(64, activation = "relu")

    tower_1 = ff1(inputs[:, 0, None]) 
    tower_1 = ff2(tower_1)
    # tower_1 = ff3(tower_1)    
    # tower_1 = keras.layers.MaxPool1D(pool_size = 50, padding='same')(tower_1)

    tower_2 = ff1(inputs[:, 1, None])
    tower_2 = ff2(tower_2)
    # tower_2 = ff3(tower_2)    
    # tower_2 = keras.layers.MaxPool1D(pool_size = 50, padding='same')(tower_2)

    tower_3 = ff1(inputs[:, 2, None])
    tower_3 = ff2(tower_3)    
    # tower_3 = ff3(tower_3)
    # tower_3 = keras.layers.MaxPool1D(pool_size = 50,  padding='same')(tower_3)

    tower_4 = ff1(inputs[:, 25, None]) 
    tower_4 = ff2(tower_4)    
    # tower_4 = ff3(tower_4)
    # tower_4 = keras.layers.MaxPool1D(pool_size = 50, padding='same')(tower_4)

    tower_5 = ff1(inputs[:, 26, None])
    tower_5 = ff2(tower_5)    
    # tower_5 = ff3(tower_5)
    # tower_5 = keras.layers.MaxPool1D(pool_size = 50, padding='same')(tower_5)

    tower_6 = ff1(inputs[:, 27, None])
    tower_6 = ff2(tower_6)    
    # tower_6 = ff3(tower_6)
    # tower_6 = keras.layers.MaxPool1D(pool_size = 50,  padding='same')(tower_6)

    merged = keras.layers.concatenate([tower_1, tower_2, tower_3, tower_4, tower_5, tower_6], axis=1)
    # merged = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=1)
    merged = keras.layers.MaxPool1D(pool_size = 3,  padding='same')(merged)


    merged = keras.layers.Dense(32, activation = "tanh")(merged)
    merged = keras.layers.Dense(32, activation = "tanh")(merged)
    merged = keras.layers.Flatten()(merged)

    out = keras.layers.Dense(32, activation='tanh')(merged)

    output = keras.layers.Dense(units=3, activation = 'tanh')(out)

    #rescale output
    output = output*tf.constant([15., 15., 15.])

    model = tf.keras.Model(inputs,output)

    return model

def Net(**kwargs):

    ''' Simple feedforward network
    '''
    #DO MAX POOLING FOR insize//2 since we are looking at two seperate point clouds!!!!!

    insize = 100 #512

    inputs = keras.Input(shape=(insize, 3)) 


    # #old way (inidividual weights)-----------------------------
    # X = keras.layers.BatchNormalization()(inputs)
    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # #----------------------------------------------------------

    #new way- use conv layers to auto share weights--------------
    X = tf.expand_dims(inputs, -1)

    # #TEST 7/28/22 -
    # #~~~~~~~
    # X = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # #~~~~~~~

    X = tf.keras.layers.Conv2D(64, [1,3], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    #TODO -> figure out why this is better than 1d conv
    #This was at 512, I dropped it to increase batch size
    X = tf.keras.layers.Conv2D(256, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = tf.reshape(X, [-1,insize,256])

    #worse than 2d...
    # X = tf.keras.layers.Conv1D(512, kernel_size = 1, strides = 1, padding = 'valid', activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    #------------------------------------------------------------

    # #NEW 7/28/22
    # # try doing a couple FF layers between conv and maxpool to get network to share information across point clouds
    # # -----------------------------------------------------------
    # # X = tf.reshape(X, [-1,insize,256]) #??
    # X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # # -----------------------------------------------------------

    # 1D Max Pooling 
    X = keras.layers.MaxPool1D(pool_size = int(insize/2))(X)
    
    #just ff -------------------------------------------------------------------------- 
    X = keras.layers.Flatten()(X)
    # X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    #----------------------------------------------------------------------------------

    # # using conv layers----------------------------------------------------------------
    # X = tf.transpose(X, [0, 2, 1]) #test - I think this is needed to perform conv on correct axis
    # # X = keras.layers.Permute((2,1))(X) #also works

    # #~~~~~~~~~~~~~~~~~~~~~~~~
    # #NEW - Need FF layer to re-arrange points so that convolutional kernels can actually do their job??
    # X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)

    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # #~~~~~~~~~~~~~~~~~~~~~~~~

    # #conv layers help 1d a lot
    # X = keras.layers.Conv1D(filters = 4, kernel_size = 8, strides = 4, padding = 'valid', activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Conv1D(filters = 2, kernel_size = 4, strides = 2, padding = 'valid', activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # # X = keras.layers.Conv1D(filters = 2, kernel_size = 3, strides = 2, padding = 'valid', activation = 'relu')(X)
    # # X = keras.layers.BatchNormalization()(X)
    # # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Flatten()(X)    
    # #----------------------------------------------------------------------------------
    
    X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    output = keras.layers.Dense(units=3, activation = 'tanh')(X) #translation only
    # output = keras.layers.Dense(units=6, activation = 'tanh')(X) #translation + rotation


    #rescale output
    # output = output*tf.constant([15., 15., 0.03]) #was this for simple models
    output = output*tf.constant([30., 30., 3.]) #increased vel using real cars
    # output = output*tf.constant([5., 5., 5.]) #KITTI

    #toilet benchmark
    # output = output*tf.constant([3., 3., 3., 3., 3., 3.])
    # output = output*tf.constant([3., 3., 3.])



    model = tf.keras.Model(inputs,output)

    return model


# def Net(**kwargs):

#     ''' Matt's network design (updated 5/16) 
#     '''

#     # insize =  50 #for default KITTInet/ FORDnet
#     insize = 100 #test with more sample points 

#     inputs = keras.Input(shape=(insize, 3)) 

#     #try to force random shuffle of input data within each batch
#     # why is this killing everything?????
#     # X = tf.transpose(inputs, [1,0,2])
#     # X = tf.random.shuffle(X)
#     # X = tf.transpose(X, [1,0,2])
#     # X = keras.layers.BatchNormalization()(X)

#     X = keras.layers.BatchNormalization()(inputs)
#     X = keras.layers.Dense(units = 64, activation = 'relu')(X)
#     X = layers.Dropout(0.2)(X)

#     # new ~~
#     X = keras.layers.BatchNormalization()(X)
#     X = keras.layers.Dense(units = 128, activation = 'relu')(X)
#     # X = layers.Dropout(0.2)(X)

#     X = keras.layers.BatchNormalization()(X)
#     X = keras.layers.Dense(units = 256, activation = 'relu')(X)
#     # X = layers.Dropout(0.2)(X)
#     # ~~~~~~

#     X = keras.layers.BatchNormalization()(X)
#     X = keras.layers.Dense(units = 512, activation = 'relu')(X)
#     # X = layers.Dropout(0.2)(X)

#     # 2D Max Pooling -------------------------------------------------------------------
#     # X = tf.keras.layers.Reshape((insize, 8, 8))(X)
#     # X = tf.keras.layers.MaxPooling2D(pool_size = (25,1), strides = None, padding = 'same')(X)
#     #test
#     # X = tf.keras.layers.Conv2D( filters = 16, kernel_size = (2)  )(X)

#     #----------------------------------------------------------------------------------


#     # 1D Max Pooling ------------------------------------------------------------------
#     X = keras.layers.MaxPool1D(pool_size = 25)(X)

#     X = tf.transpose(X, [0, 2, 1]) #test - I think this is needed to perform conv on correct axis
#     # X = keras.layers.Permute((2,1))(X) #also works

#     #conv layers help 1d a lot
#     X = keras.layers.Conv1D(filters = 32, kernel_size = 3, strides = 3, padding = 'valid')(X)
#     X = keras.layers.BatchNormalization()(X)
#     X = keras.layers.Dense(units = 64, activation = 'relu')(X)
#     X = keras.layers.BatchNormalization()(X)
#     # X = layers.Dropout(0.1)(X)

#     # #test repeat
#     X = keras.layers.Conv1D(filters = 32, kernel_size = 5, strides = 5, padding = 'valid')(X)
#     X = keras.layers.BatchNormalization()(X)
#     X = keras.layers.Dense(units = 64, activation = 'relu')(X)
#     X = keras.layers.BatchNormalization()(X)
#     # X = layers.Dropout(0.1)(X)


#     X = keras.layers.Conv1D(filters = 32, kernel_size = 8, strides = 8, padding = 'valid')(X)
#     X = keras.layers.BatchNormalization()(X)
#     X = keras.layers.Dense(units = 64, activation = 'relu')(X)
#     X = keras.layers.BatchNormalization()(X)
#     # X = layers.Dropout(0.1)(X)
#     #----------------------------------------------------------------------------------

#     X = keras.layers.Flatten()(X)    

#     X = keras.layers.Dense(units = 256, activation = 'relu')(X)
#     # X = layers.Dropout(0.5)(X)
#     X = keras.layers.BatchNormalization()(X)
#     # X = layers.Dropout(0.2)(X)
    
#     X = keras.layers.Dense(units = 128, activation = 'relu')(X)
#     X = keras.layers.BatchNormalization()(X)
#     # X = layers.Dropout(0.2)(X)


#     X = keras.layers.Dense(units = 64, activation = 'relu')(X)
#     X = layers.Dropout(0.2)(X)
#     X = keras.layers.BatchNormalization()(X)
    
#     output = keras.layers.Dense(units=3, activation = 'tanh')(X)

#     #rescale output
#     # output = output*tf.constant([15., 15., 0.03]) #was this for simple models
#     # output = output*tf.constant([30., 30., 3.]) #increased vel using real cars
#     output = output*tf.constant([3., 3., 0.3]) #KITTI
#     # output = output*tf.constant([5., 5., 0.5])


#     model = tf.keras.Model(inputs,output)

#     return model

def PointNet(*kwargs):

    ''' attempt to solve the permuation invarinace problem copying strategy used by PointNet
    https://keras.io/examples/vision/pointnet_segmentation/ '''

    inputs = keras.Input(shape=(50, 3)) 

    X = tnet(inputs, 3)
    X = conv_bn(X, 32)
    # X = conv_bn(X, 32)
    X = tnet(X, 32)
    X = conv_bn(X, 32)
    # X = conv_bn(X, 64)
    # X = conv_bn(X, 512)
    X = layers.GlobalMaxPooling1D()(X)
    # X = dense_bn(X, 256)
    # X = layers.Dropout(0.3)(X) #0.3
    X = dense_bn(X, 128)
    # X = layers.Dropout(0.3)(X) #0.3


    #scale output to match scale of motion in data
    output = keras.layers.Dense(units=3, activation = 'tanh')(X)
    output = output*tf.constant([15., 15., 0.03])

    model = tf.keras.Model(inputs,output)

    return model

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64) #64
    x = conv_bn(x, 512) #512
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256) #256
    x = dense_bn(x, 128) #128
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))