import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

#TODO: figure out whats going on with conv1D not adjusting output shape

def PCRnet(**kwargs):

    ''' Another attempt at permutation invariance, this time using PCRnet architecture '''

    insize =  50 #200 
    #init shared fully connected and conv layers
    ff1 = keras.layers.Dense(256, activation = "relu")
    ff2 = keras.layers.Dense(512, activation = "relu")
    # ff22 = keras.layers.Dense(2048, activation = "relu")

    conv1 = keras.layers.Conv1D(filters = 32, kernel_size = 3) #, padding = 'same')
    conv2 = keras.layers.Conv1D(filters = 32, kernel_size = 8, strides = 4) #, padding = 'same')
    ff3 = keras.layers.Dense(64, activation = "relu")
    ff4 = keras.layers.Dense(64, activation = "relu")


	#get input    
    inputs = keras.Input(shape=(insize, 3)) 

    #split into two "MLPs", one for each set of input sample points
    X1 = inputs[:,:(insize//2)]
    X1 = keras.layers.BatchNormalization()(X1)
    X2 = inputs[:,(insize//2):]
    X2 = keras.layers.BatchNormalization()(X2)

    #apply siamese feedforward layers-----------------------------------------
    X1 = ff1(X1)
    X1 = keras.layers.BatchNormalization()(X1)
    # X2 = layers.Dropout(0.2)(X2) #0.3

    X2 = ff1(X2)
    X2 = keras.layers.BatchNormalization()(X2)
    # X2 = layers.Dropout(0.2)(X2) #0.3

    X1 = ff2(X1)
    X1 = keras.layers.BatchNormalization()(X1)
    X2 = ff2(X2)
    X2 = keras.layers.BatchNormalization()(X2)

    # X1 = ff22(X1)
    # X1 = keras.layers.BatchNormalization()(X1)
    # X2 = ff22(X2)
    # X2 = keras.layers.BatchNormalization()(X2)
    #-------------------------------------------------------------------------

    # #apply symmetric functions (MaxPool1D) to each tower to remove any permutation invariance ----------
    X1 = keras.layers.MaxPool1D(pool_size = (25))(X1)
    X2 = keras.layers.MaxPool1D(pool_size = (25))(X2)
    # #---------------------------------------------------------------------------------------------------

    # # 3D conv on each feature representation----------------------------------
    # #reshape to true 3d data representation
    # X1 = keras.layers.Reshape([8, 8, 8, 1])(X1) 
    # X2 = keras.layers.Reshape([8, 8, 8, 1])(X2) 

    # X1 = keras.layers.Conv3D(filters = 32, kernel_size = 2, strides = (1,1,1), padding = 'valid')(X1)
    # X1 = keras.layers.BatchNormalization()(X1)
    # X1 = ff3(X1)
    # X1 = keras.layers.BatchNormalization()(X1)

    # X2 = keras.layers.Conv3D(filters = 32, kernel_size = 2, strides = (1,1,1), padding = 'valid')(X2)
    # X2 = keras.layers.BatchNormalization()(X2)
    # X2 = ff3(X2)
    # X2 = keras.layers.BatchNormalization()(X2)


    # X1 = keras.layers.Conv3D(filters = 32, kernel_size = 3, strides = (1,1,1), padding = 'valid')(X1)
    # X1 = keras.layers.BatchNormalization()(X1)
    # X1 = ff4(X1)
    # X1 = keras.layers.BatchNormalization()(X1)

    # X2 = keras.layers.Conv3D(filters = 32, kernel_size = 3, strides = (1,1,1), padding = 'valid')(X2)
    # X2 = keras.layers.BatchNormalization()(X2)
    # X2 = ff4(X2)
    # X2 = keras.layers.BatchNormalization()(X2)


    # #-------------------------------------------------------------------------


    #reshape and perform 1D conv on each feature representation---------------
    X1 = tf.transpose(X1, [0, 2, 1])
    X2 = tf.transpose(X2, [0, 2, 1])

    X1 = conv1(X1)
    X1 = keras.layers.BatchNormalization()(X1)
    X1 = ff3(X1)
    X1 = keras.layers.BatchNormalization()(X1)

    X2 = conv1(X2)
    X2 = keras.layers.BatchNormalization()(X2)
    X2 = ff3(X2)
    X2 = keras.layers.BatchNormalization()(X2)

    X1 = conv2(X1)
    X1 = keras.layers.BatchNormalization()(X1)
    X1 = ff4(X1)
    X1 = keras.layers.BatchNormalization()(X1)

    X2 = conv2(X2)
    X2 = keras.layers.BatchNormalization()(X2)
    X2 = ff4(X2)
    X2 = keras.layers.BatchNormalization()(X2)
    #-------------------------------------------------------------------------

    #Flatten and concatenate -------------------------------------------------
    X1 = keras.layers.Flatten()(X1)
    X2 = keras.layers.Flatten()(X2)
    X = keras.layers.Concatenate(axis=1)([X1, X2])

    # more fully connected layers to perform the matching operation on encoded data -------
    # X = keras.layers.Dense(512, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(128, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    # X = layers.Dropout(0.2)(X) #0.3


    #--------------------------------------------------------------------------------------    


    output = keras.layers.Dense(3, activation = 'tanh')(X)


    #rescale output
    # output = output*tf.constant([15., 15., 0.03]) #was this for simple prism models
    output = output*tf.constant([30., 30., 3.]) #increased vel and using more realistic 3d shapes (cars, busses, etc)

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

    insize = 50 # 200 #50

    inputs = keras.Input(shape=(insize, 3)) 

    #try to force random shuffle of input data within each batch
    # why is this killing everything?????
    # X = tf.transpose(inputs, [1,0,2])
    # X = tf.random.shuffle(X)
    # X = tf.transpose(X, [1,0,2])
    # X = keras.layers.BatchNormalization()(X)

    X = keras.layers.BatchNormalization()(inputs)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)

    # new ~~
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 128, activation = 'relu')(X)

    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    # ~~~~~~

    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 512, activation = 'relu')(X)

    # 2D Max Pooling -------------------------------------------------------------------
    # X = tf.keras.layers.Reshape((insize, 8, 8))(X)
    # X = tf.keras.layers.MaxPooling2D(pool_size = (25,1), strides = None, padding = 'same')(X)
    #test
    # X = tf.keras.layers.Conv2D( filters = 16, kernel_size = (2)  )(X)

    #----------------------------------------------------------------------------------


    # 1D Max Pooling ------------------------------------------------------------------
    X = keras.layers.MaxPool1D(pool_size = 25)(X)

    X = tf.transpose(X, [0, 2, 1]) #test - I think this is needed to perform conv on correct axis
    # X = keras.layers.Permute((2,1))(X) #also works

    #conv layers help 1d a lot
    X = keras.layers.Conv1D(filters = 32, kernel_size = 3, padding = 'valid')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    # #test repeat
    X = keras.layers.Conv1D(filters = 32, kernel_size = 5, padding = 'valid')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)


    X = keras.layers.Conv1D(filters = 32, kernel_size = 8, strides = 3, padding = 'valid')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    #----------------------------------------------------------------------------------

    X = keras.layers.Flatten()(X)    

    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    

    output = keras.layers.Dense(units=3, activation = 'tanh')(X)

    #rescale output
    # output = output*tf.constant([15., 15., 0.03]) #was this for simple models
    # output = output*tf.constant([30., 30., 3.]) #increased vel using real cars
    output = output*tf.constant([3., 3., 0.3]) #KITTI


    model = tf.keras.Model(inputs,output)

    return model

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