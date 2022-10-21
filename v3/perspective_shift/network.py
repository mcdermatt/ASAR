import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

# #for using PointNet package, need to run py39 conda env ________
# import tensorflow_graphics as tfg
# from tensorflow_graphics.nn.layer.pointnet import PointNetConv2Layer

# def PointNet(**kwargs):
#     """ Using PointNet 2D conv layer provided by ___ """

#     insize = 100
#     inputs = keras.Input(shape=(insize, 3))

#     X = test_conv2(64, inputs, 1, True)

#     output = keras.layers.Dense(units=3, activation = 'tanh')(X) #translation only
#     output = output*tf.constant([5., 5., 5.]) #KITTI
#     model = tf.keras.Model(inputs,output)

#     return model

# def test_conv2(input_shape, channels, momentum, training):
#     B, N, X, _ = input_shape
#     inputs = tf.random.uniform(input_shape)
#     layer = PointNetConv2Layer(channels, momentum)
#     outputs = layer(inputs, training=training)
#     assert outputs.shape == (B, N, X, channels)
# #_________________________________________________________________

def Attention(**kwargs):
    """ Trying out attention network to replace PointNet-style encoding of input features"""
    insize = 100

    inputs = keras.Input(shape=(insize, 3)) 

    X = tf.expand_dims(inputs, -1)
    X = tf.keras.layers.Conv2D(128, [1,3], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = tf.reshape(X, [-1,insize,64])

    X = tf.keras.layers.Attention()([X, X])
    X = keras.layers.MaxPool1D(pool_size = int(insize/2))(X)

    X = keras.layers.Flatten()(X)

    X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 128, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)


    output = keras.layers.Dense(units=3, activation = 'tanh')(X) #translation only
    output = output*tf.constant([5., 5., 5.]) #KITTI
    model = tf.keras.Model(inputs,output)

    return model

def FFNet(**kwargs):

    insize = 200

    inputs = keras.Input(shape=(insize, 3)) 
    X = keras.layers.Flatten()(inputs)

    
    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)

    # X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)


    # X = keras.layers.Dense(units = 2045, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 2048, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 2048, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)


    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)



    output = keras.layers.Dense(units=3, activation = 'tanh')(X) #translation only
    output = output*tf.constant([5., 5., 5.]) #rescale output
    model = tf.keras.Model(inputs,output)

    return model

def LUT_test(**kwargs):

    ''' Test network for estimating LUT
    '''
    insize = 108
    inputs = keras.Input(shape=(insize, 3)) 

    # #simple MLP network -----------------------------------------------
    # X = keras.layers.Flatten()(inputs)
    # X = keras.layers.Dense(units = 2048, activation = 'relu')(X)
    # # X = keras.layers.BatchNormalization()(X)

    # X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    # # X = keras.layers.BatchNormalization()(X)

    # X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # #------------------------------------------------------------------

    X = tf.expand_dims(inputs[:,:50], -1) #don't actually need scan 2 for this test
    # bounds = inputs[:,100:] #if not doing pointnet on bounds


    #PointNet layers on points from scan1 -----------------------------------------------------------
    X = tf.keras.layers.Conv2D(64, [1,3], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(128, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.MaxPool2D([50, 1])(X)
    #------------------------------------------------------------------------------------------------

    X = keras.layers.Flatten()(X)

    #Run PointNet on bounds--------------------------------------------------------------------------
    bounds = tf.expand_dims(inputs[:,100:], -1)
    # bounds = tf.transpose(bounds, [0, 2, 1])
    bounds = tf.keras.layers.Conv2D(64, [1,3], padding = 'valid', strides = [1,1], activation = 'relu')(bounds)
    bounds = keras.layers.BatchNormalization()(bounds)

    bounds = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(bounds)
    bounds = keras.layers.BatchNormalization()(bounds)

    bounds = keras.layers.MaxPool2D([8, 1])(bounds)
    bounds = keras.layers.Flatten()(bounds)
    #------------------------------------------------------------------------------------------------

    X = keras.layers.concatenate([X, bounds])

    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X) #skip batch norm on first layer??

    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    #for estimating LUT
    output = keras.layers.Dense(units = 9, activation = 'tanh')(X) 
    output = tf.reshape(output, (-1,3,3))
    model = tf.keras.Model(inputs,output)

    return model

def TestNet(**kwargs):

    ''' Simple feedforward network
    '''
    #DO MAX POOLING FOR insize//2 since we are looking at two seperate point clouds!!!!!

    insize = 200 #1024 #512

    inputs = keras.Input(shape=(insize, 3)) 


    # #old way (inidividual weights)-----------------------------
    # X = keras.layers.BatchNormalization()(inputs)
    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # #----------------------------------------------------------

    #new way- use conv layers to auto share weights--------------
    X = tf.expand_dims(inputs, -1)


    # #test using PointNet layers------------
    # momentum =  0.99 #0.1 #TODO: figure out what works best here??
    # X = PointNetConv2Layer(64, momentum)(X)
    # X = PointNetConv2Layer(64, momentum)(X)
    # X = PointNetConv2Layer(256, momentum)(X)
    # #--------------------------------------

    #was this-------------
    # code for PCR-NET says to start with [1,3] kernel, however, code in learning3d (also published by Sarode) starts with 1x1
    X = tf.keras.layers.Conv2D(256, [1,3], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(256, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(512, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    # # TODO -> figure out why this is better than 1d conv
    # This was at 512, I dropped it to increase batch size
    X = tf.keras.layers.Conv2D(1024, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    #------------------------------------------------------------

    #worse than 2d...
    # X = tf.keras.layers.Conv1D(512, kernel_size = 1, strides = 1, padding = 'valid', activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)

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
    # X = tf.reshape(X, [-1,insize,64])
    # X = keras.layers.MaxPool1D(pool_size = int(insize/2))(X)

    # 2D Max Pooling - used by author of PointNet, not by PCR-Net(?)
    X = keras.layers.MaxPool2D([insize//2, 1])(X)
    # X = keras.layers.AveragePooling2D([insize//2, 1])(X) #TODO: test this
    
    #just ff -------------------------------------------------------------------------- 
    X = keras.layers.Flatten()(X)
    # X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    #----------------------------------------------------------------------------------

    # using conv layers----------------------------------------------------------------
    # X = tf.transpose(X, [0, 2, 1]) #test - I think this is needed to perform conv on correct axis
    # X = keras.layers.Permute((2,1))(X) #also works

    # #~~~~~~~~~~~~~~~~~~~~~~~~
    ## Need FF layer to re-arrange points so that convolutional kernels can actually do their job??
    # X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)

    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # #~~~~~~~~~~~~~~~~~~~~~~~~

    #conv layers help 1d a lot
    X = tf.expand_dims(X, -1)
    X = keras.layers.Conv1D(filters = 64, kernel_size = 8, strides = 4, padding = 'valid', activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Conv1D(filters = 64, kernel_size = 4, strides = 4, padding = 'valid', activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Conv1D(filters = 64, kernel_size = 4, strides = 4, padding = 'valid', activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Flatten()(X)    
    #----------------------------------------------------------------------------------
    
    # X = keras.layers.Dense(units = insize, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    #was 256
    X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)


    #translation only
    output = keras.layers.Dense(units=3, activation = 'tanh')(X)
    output = output*tf.constant([5., 5., 5.]) #rescale output

    # #6DOF (for toilet benchmark)
    # output = keras.layers.Dense(units=6, activation = 'tanh')(X)
    # output = output*tf.constant([3., 3., 3., 3., 3., 3.])



    model = tf.keras.Model(inputs,output)

    return model

def PointNet(**kwargs):

    """ debug using Tensorflow Graphics PointNetConv2 layers """

    insize = 512
    inputs = keras.Input(shape=(insize,3))

    X = tf.expand_dims(inputs, -1)


    momentum =  0.99 #0.1 #TODO: figure out what works best here??
    X = PointNetConv2Layer(64, momentum)(X)
    X = PointNetConv2Layer(64, momentum)(X)
    X = PointNetConv2Layer(256, momentum)(X)


    X = keras.layers.MaxPool2D([insize//2, 1])(X)

    X = keras.layers.Flatten()(X)

    #temp
    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)


    #6DOF (for toilet benchmark)
    output = keras.layers.Dense(units=6, activation = 'tanh')(X)
    output = output*tf.constant([3., 3., 3., 3., 3., 3.])
    model = tf.keras.Model(inputs,output)
    return model



def Net(**kwargs):

    ''' Simple feedforward network
    '''
    #DO MAX POOLING FOR insize//2 since we are looking at two seperate point clouds!!!!!

    insize = 200 #1024 #512 #100 #200

    inputs = keras.Input(shape=(insize, 3)) 


    # #old way (inidividual weights)-----------------------------
    # X = keras.layers.BatchNormalization()(inputs)
    # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)
    # X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    # #----------------------------------------------------------

    #new way- use conv layers to auto share weights--------------
    X = tf.expand_dims(inputs, -1)


    # #test using PointNet layers------------
    # momentum =  0.99 #0.1 #TODO: figure out what works best here??
    # X = PointNetConv2Layer(64, momentum)(X)
    # X = PointNetConv2Layer(64, momentum)(X)
    # X = PointNetConv2Layer(256, momentum)(X)
    # #--------------------------------------

    #was this-------------
    # code for PCR-NET says to start with [1,3] kernel, however, code in learning3d (also published by Sarode) starts with 1x1
    X = tf.keras.layers.Conv2D(256, [1,3], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(256, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(512, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    # # TODO -> figure out why this is better than 1d conv
    # This was at 512, I dropped it to increase batch size
    X = tf.keras.layers.Conv2D(1024, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    #------------------------------------------------------------

    #worse than 2d...
    # X = tf.keras.layers.Conv1D(512, kernel_size = 1, strides = 1, padding = 'valid', activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)

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
    # X = tf.reshape(X, [-1,insize,64])
    # X = keras.layers.MaxPool1D(pool_size = int(insize/2))(X)

    # 2D Max Pooling - used by author of PointNet, not by PCR-Net(?)
    X = keras.layers.MaxPool2D([insize//2, 1])(X)
    # X = keras.layers.AveragePooling2D([insize//2, 1])(X) #TODO: test this
    
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

    # # #~~~~~~~~~~~~~~~~~~~~~~~~
    # ## Need FF layer to re-arrange points so that convolutional kernels can actually do their job??
    # # X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    # # X = keras.layers.BatchNormalization()(X)

    # # X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    # # X = keras.layers.BatchNormalization()(X)
    # # #~~~~~~~~~~~~~~~~~~~~~~~~

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
    
    # X = keras.layers.Dense(units = insize, activation = 'relu')(X)
    # X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 1024, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    #was 256
    X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)


    #translation only
    output = keras.layers.Dense(units=3, activation = 'tanh')(X)
    output = output*tf.constant([5., 5., 5.]) #rescale output

    # #6DOF (for toilet benchmark)
    # output = keras.layers.Dense(units=6, activation = 'tanh')(X)
    # output = output*tf.constant([3., 3., 3., 3., 3., 3.])



    model = tf.keras.Model(inputs,output)

    return model

def bestNet(**kwargs):

    '''  best network structure so far 7/31/22 '''
    
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



def smallNet(**kwargs):

    ''' Simple feedforward network
    '''
    #DO MAX POOLING FOR insize//2 since we are looking at two seperate point clouds!!!!!

    insize = 200 #1024 #512 #100 #200

    inputs = keras.Input(shape=(insize, 3)) 

    #use conv layers to share weights
    X = tf.expand_dims(inputs, -1)

    # code for PCR-NET says to start with [1,3] kernel, however, code in learning3d (also published by Sarode) starts with 1x1
    X = tf.keras.layers.Conv2D(64, [1,3], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = tf.keras.layers.Conv2D(64, [1,1], padding = 'valid', strides = [1,1], activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.MaxPool2D([insize//2, 1])(X)    
    X = keras.layers.Flatten()(X)

    X = keras.layers.Dense(units = 512, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 256, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dense(units = 64, activation = 'relu')(X)
    X = keras.layers.BatchNormalization()(X)


    #translation only
    output = keras.layers.Dense(units=3, activation = 'tanh')(X)
    output = output*tf.constant([5., 5., 5.]) #rescale output

    # #6DOF (for toilet benchmark)
    # output = keras.layers.Dense(units=6, activation = 'tanh')(X)
    # output = output*tf.constant([3., 3., 3., 3., 3., 3.])


    model = tf.keras.Model(inputs,output)

    return model