from vedo import *
from utils import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import pickle
import matplotlib.pyplot as plt

#limit GPU memory ------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    memlim = 4*1024
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memlim)])
  except RuntimeError as e:
    print(e)
#-----------------------------------------------------------------


def v2t(vector):
    """converts a transformation vector to homogenous coordinate system"""
    if len(tf.shape(vector)) == 1: #allow for 1-D or N-D input
        vector = vector[None,:]
    angs = vector[:,3:]
    phi = angs[:,0]
    theta = angs[:,1]
    psi = angs[:,2]
    rot = tf.Variable([[cos(theta)*cos(psi), sin(psi)*cos(phi) + sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi)],
                       [-sin(psi)*cos(theta), cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(psi) + sin(theta)*sin(psi)*cos(phi)],
                       [sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta)]])
    rot = tf.transpose(rot, [2, 0, 1])
    trans = vector[:,:3]
    trans = np.reshape(trans, (np.shape(rot)[0], 3, 1))
    transform = tf.concat((rot, trans), axis = -1)
    extra = tf.tile(tf.constant([[[0., 0., 0., 1.]]], dtype = tf.double), (np.shape(rot)[0],1,1))
    transform = tf.concat((transform, extra), axis = -2)
    return transform

def t2v(mat):
    """converts transformation matrix to state vector"""
    if len( tf.shape(mat) ) == 2:
        mat = mat[None, :, :]
    R_sum = np.sqrt(( mat[:,0,0]**2 + mat[:,0,1]**2 + mat[:,1,2]**2 + mat[:,2,2]**2 ) / 2)
    phi = np.arctan2(-mat[:,1,2],mat[:,2,2])
    theta = np.arctan2(mat[:,0,2], R_sum)
    psi = np.arctan2(-mat[:,0,1], mat[:,0,0])
    angs = np.array([phi, theta, psi])
    vector = tf.concat((mat[:,:3,-1], angs.T), axis =1)
    return vector

def get_J(e, ij):
    """Forms sparse jacobian J, with entries A and B at indices i and j 
    J == [N, 6, 6*N], N = number of nodes
    e == [N, 6] error vectors
    ij == [N, 2] contains scan indices of each node ex: [scan2, scan5]
    """
    total_num_of_nodes = np.max(ij) + 1 #TODO: is this too big??
    if len(tf.shape(ij))< 2: #expand dimensions if only single pair passed in
        ij = ij[None,:]
    A_ij, B_ij = get_A_ij_B_ij(e)
    
    # Need to tile DIFFERENT AMOUNT depending on the index 
    #    TODO: move to batch operation?
    J = tf.zeros([0,6,total_num_of_nodes*6])
    for k in range(len(ij)):
        #TODO: add logic for when i and j are same value, when j = i + 1 ...
        leading = tf.tile(tf.zeros([6,6]), [1, ij[k,0] ]) #leading zeros before i
#         print("\n leading \n", leading)
        between = tf.tile(tf.zeros([6,6]), [1, ij[k,1] - ij[k,0] - 1 ]) #zeros between i and j
#         print("\n between: \n", between)
        ending  = tf.tile(tf.zeros([6,6]), [1, total_num_of_nodes - ij[k,1] - 1 ])
        J_ij = tf.concat([leading, A_ij[k], between, B_ij[k], ending], axis = 1)
        # print("\n J(", k, ") \n", J_ij)      #for debug  
        J = tf.concat((J, J_ij[None,:,:]), axis = 0)

    # # test - apply constraint within J??
    # J = J.numpy()
    # J[0] += np.append(np.eye(6),  np.zeros([6,6*len(ij)]), axis = 1)
    # J = tf.convert_to_tensor(J)

    return J

def get_information_matrix(pred_stds):
    """returns information matrix (omega) from ICET cov_estimates"""

#     #I think this is wrong ... 
#     pred_stds = tf.convert_to_tensor(pred_stds)[:,:,None] #convert to TF Tensor
#     cov = pred_stds @ tf.transpose(pred_stds, (0,2,1))    #convert predicted stds -> covariance matrix
#     info = tf.linalg.pinv(cov) #invert
    
    #debug - set to identity
    info = tf.tile(tf.eye(6)[None,:,:], [tf.shape(pred_stds)[0] , 1, 1])
    info = tf.cast(info, tf.double)

#     #debug - weigh rotations more heavily than translations
# #     m = tf.linalg.diag(tf.constant([1., 1., 1., 10., 10., 10.]))
# #     m = tf.linalg.diag(tf.constant([10., 10., 10., 1., 1., 1.])) #vice-versa
#     m = tf.linalg.diag(tf.constant([100., 100., 100., 100., 100., 100.])) #huge values
#     info = tf.tile(m[None,:,:], [tf.shape(pred_stds)[0] , 1, 1])
#     info = tf.cast(info, tf.double)


    return info

def get_b(e, omega, J):
    """gets b matrix, using batched operations """

    b = tf.math.reduce_sum(tf.cast(tf.transpose(J, (0,2,1)), tf.double) @ omega @ e, axis = 0)
    # print("\n b: \n", tf.shape(b))

    return b

def get_ij(ij_raw):
    """generates ij matrix, which describes which nodes are connected to 
       each other through odometry constraints. 
       Removes skipped indices and shifts everything to start at 0"""
#     print("ij_raw: \n", ij_raw)    
    y, idx = tf.unique(tf.reshape(ij_raw, [-1]))
#     print("\n y: \n", y, "\n idx: \n", idx)    
    ij = tf.reshape(idx, [-1,2])
    return ij

def get_H(J, omega):
    """returns hessian H"""

    # print("\n J: \n", tf.shape(J))
    # print("\n omega: \n", tf.shape(omega))

    H_ij = tf.transpose(J, (0,2,1)) @ tf.cast(omega, tf.float32) @ J
    H = tf.math.reduce_sum(H_ij, axis = 0)

    #produces negative eigenvals if we don't fix first point in chain
    # print("\n H before constraint: \n", tf.shape(H))
    # print("\n eigval H before constraint:\n", tf.linalg.eigvals(H))

    # constrain_11 = tf.pad(tf.eye(6), [[0,len(H)-6],[0,len(H)-6]]) #was this
    constrain_11 = tf.pad(tf.eye(6), [[1,len(H)-7],[1,len(H)-7]]) #test
    # constrain_11 = tf.pad(len(J) * tf.eye(6), [[0,len(H)-6],[0,len(H)-6]]) #test
    # print("constrain_11: \n", constrain_11)
    H = H + constrain_11
    # print("\n eigval H after constraint:\n", tf.linalg.eigvals(H))
#     H = tf.convert_to_tensor(np.tril(H.numpy()).T + np.tril(H.numpy(),-1)) #force H to be symmetric
    return H

def get_X(x, ij):
    """given x, a list of global poses, this function returns 
       the relative transformations X, that describe the same relationships described by the constraints z
       x  -> global poses of each transform
       ij -> indices of first and second element of x being considered
       """

    #get transform of fist elements in each pair, ordered by how they appear in ij
    first_vec = tf.gather(x, ij[:,0])
    second_vec = tf.gather(x, ij[:,1])

    first_tensor = v2t(tf.cast(first_vec, tf.double))
    second_tensor = v2t(tf.cast(second_vec, tf.double))

    #represents pose of x in 2nd node relative to pose in 1st
    
    #Problem with this answer is that this sets first tensor w.r.t. the world axis 
    X = tf.linalg.pinv(first_tensor) @ second_tensor #was this
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #TODO: 
    # loop through all past elements [0 ... i] of x and
    # iteratively apply transformations in 

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



    return X

def get_A_ij_B_ij(e_ij):
    """calculates nonzero terms from the Jacobian of error function w.r.t. nodes i and j using TensorFlow
        e_ij == error function [x, y, z, phi, theta, psi]
        
        NOTE: this works with batch operations: error vectors passed in as tensor will result in
                corresponding output in the same order 
    """
    e_ij = tf.cast(e_ij, tf.float32)
    p_point = e_ij[:,:3]
    phi = e_ij[:,3][:,None]
    theta = e_ij[:,4][:,None]
    psi = e_ij[:,5][:,None]
    
    eyes = tf.tile(-tf.eye(3)[None,:,:], [tf.shape(p_point)[0] , 1, 1]) #was this
#     eyes = tf.tile(tf.eye(3)[None,:,:], [tf.shape(p_point)[0] , 1, 1]) #test
    
#     deriv of R() wrt phi
    dRdPhi = tf.Variable([[tf.zeros(len(phi), dtype = phi.dtype)[:,None], (-sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi)), (cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi))],
                       [tf.zeros(len(phi), dtype = phi.dtype)[:,None], (-sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)), (cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi))], 
                       [tf.zeros(len(phi), dtype = phi.dtype)[:,None], (-cos(phi)*cos(theta)), (-sin(phi)*cos(theta))] ])[:,:,:,0]
    dRdPhi = tf.transpose(dRdPhi, (2,0,1))
    Jx = dRdPhi @ p_point[:,:,None]
    
    # (deriv of R() wrt theta).dot(p_point)
    dRdTheta = tf.Variable([[(-sin(theta)*cos(psi)), (cos(theta)*sin(phi)*cos(psi)), (-cos(theta)*cos(phi)*cos(psi))],
                               [(sin(psi)*sin(theta)), (-cos(theta)*sin(phi)*sin(psi)), (cos(theta)*sin(psi)*cos(phi))],
                               [(cos(theta)), (sin(phi)*sin(theta)), (-sin(theta)*cos(phi))] ])[:,:,:,0]
    dRdTheta = tf.transpose(dRdTheta, (2,0,1))
    Jy = dRdTheta @ p_point[:,:,None]

    # deriv of R() wrt psi
    dRdPsi = tf.Variable([[(-cos(theta)*sin(psi)), (cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi)), (cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi)) ],
                                       [(-cos(psi)*cos(theta)), (-sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi)), (-sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi))],
                                       [tf.zeros(len(phi), dtype = phi.dtype)[:,None],tf.zeros(len(phi), dtype = phi.dtype)[:,None],tf.zeros(len(phi), dtype = phi.dtype)[:,None]]])[:,:,:,0]
    dRdPsi = tf.transpose(dRdPsi, (2,0,1))
    Jz = dRdPsi @ p_point[:,:,None]
    
    top = tf.concat([eyes, Jx, Jy, Jz], axis = 2) #was this
    flipped = tf.transpose(tf.concat([Jx, Jy, Jz], axis = 2), (0,2,1))     #was this
    
    bottom = tf.concat([-flipped, -eyes], axis = 2) #works???
#     bottom = tf.concat([flipped, -eyes], axis = 2) #test

#     top = tf.concat([eyes, tf.zeros(tf.shape(flipped))], axis = 2) #test
#     bottom = tf.concat([tf.zeros(tf.shape(flipped)), -eyes], axis = 2) #test
    
    A_ij = tf.concat([top, bottom], axis = 1) #was this
    B_ij = -A_ij #was this
#     print("\n A_ij: \n", A_ij[0])
    return A_ij, B_ij

def get_e(Zij, Xij):
    """calculates error function w.r.t. nodes i and j
    Zij == pose j relative to i according to nodes (rotation matrix)
    Xij == pose j relative to i according to constraints (rotation matrix)
    """        

    # print("\n Xij \n", tf.shape(Xij))

    # was just this ~~~~~~~~~~~~~~~~~~
    e = t2v(tf.linalg.pinv(Zij) @ Xij)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

    #loop through each element in Xij and apply sequential transformations



    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

    # print("\n e \n", tf.shape(e))
    
    return(e)    