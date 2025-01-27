from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget
import numpy as np
import tensorflow as tf
import time
import cv2
import sys
import os
current = os.getcwd()
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)
sys.path.append(parent_directory+"/point_cloud_rectification")
from ICET_spherical import ICET
from linear_corrector import LC
from scipy.interpolate import griddata

from utils import R_tf
from metpy.calc import lat_lon_grid_deltas
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import copy
import trimesh
import open3d as o3d


from pillow_heif import register_heif_opener
from matplotlib import pyplot as p
from colmapParsingUtils import *
from scipy.spatial.transform import Rotation as R
import cv2

from tqdm import tqdm_notebook as tqdm
from PIL import Image

# tf.compat.v1.enable_eager_execution()
# tf.compat.v1.disable_eager_execution() #trying this to see if more memory efficient 



def posenc(x, embed_dims):
  rets = [x]
  for i in range(embed_dims):
    for fn in [tf.sin, tf.cos]:
      rets.append(fn(2.**i * x))
  return tf.concat(rets, -1)

#2**18 is below the sensor noise threshold??
# L_embed =  5 #18 #15 #10 #6
pos_embed_dims = 15 #18 #14
rot_embed_dims = 5  #4
pos_embed_dims_coarse = 15 
rot_embed_dims_coarse = 5

embed_fn = posenc

def init_model(D=8, W=256): #10,512 produced highest resolution so far...
    relu = tf.keras.layers.LeakyReLU() #per LOC-NDF   
    dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act, kernel_initializer='glorot_uniform')

    inputs = tf.keras.Input(shape=(6 + 3*2*(rot_embed_dims) + 3*2*(pos_embed_dims))) #new (embedding dims (4) and (10) )
    outputs = inputs[:,:(3+3*2*(pos_embed_dims))] #only look at positional stuff for first few layers

    for i in range(D):
        outputs = dense()(outputs)

        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs[:,:(3+3*2*(pos_embed_dims))]], -1)
            outputs = tf.keras.layers.LayerNormalization()(outputs) #as recomended by LOC-NDF 

    #combine output of first few layers with view direction components
    combined = tf.concat([outputs, inputs[:,(3+3*2*(pos_embed_dims)):]], -1)
    combined = dense(256, act=relu)(combined) #old
    combined = dense(128, act=relu)(combined) #old
    # combined = dense(512, act=relu)(combined) #use for v11
    # combined = dense(256, act=relu)(combined) #use for v11
    # combined = dense(128, act=relu)(combined) #use for v11
    combined = dense(2, act=None)(combined)
    model = tf.keras.Model(inputs=inputs, outputs=combined)
    return model

# #DEBUG -- pretend density isn't view dependant 
# def init_model(D=8, W=256):
#     relu = tf.keras.layers.LeakyReLU() #per LOC-NDF   
#     dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act, kernel_initializer='glorot_uniform')

#     inputs = tf.keras.Input(shape=(6 + 3*2*(rot_embed_dims) + 3*2*(pos_embed_dims))) #new (embedding dims (4) and (10) )
#     sigma_channel = inputs[:,:(3+3*2*(pos_embed_dims))] #only look at positional stuff for first few layers

#     for i in range(D):
#         sigma_channel = dense()(sigma_channel)

#         if i%4==0 and i>0:
#             sigma_channel = tf.concat([sigma_channel, inputs[:,:(3+3*2*(pos_embed_dims))]], -1)
#             sigma_channel = tf.keras.layers.LayerNormalization()(sigma_channel) #as recomended by LOC-NDF 
#     #bring down to single channel
#     sigma_channel = dense(1, act=None)(sigma_channel)

#     #combine output of first few layers with view direction components
#     ray_drop_channel = tf.concat([sigma_channel, inputs[:,(3+3*2*(pos_embed_dims)):]], -1)
#     ray_drop_channel = dense(256, act=relu)(ray_drop_channel) 
#     ray_drop_channel = dense(128, act=relu)(ray_drop_channel)
#     ray_drop_channel = dense(1, act=None)(ray_drop_channel)
#     combined = tf.concat([sigma_channel, ray_drop_channel], -1)
#     model = tf.keras.Model(inputs=inputs, outputs=combined)
#     return model



def init_model_proposal(D=8, W=256):  #8, 256
    relu = tf.keras.layers.ReLU() #OG NeRF   
    leaky_relu = tf.keras.layers.LeakyReLU() #per LOC-NDF   
    # sigmoid = tf.keras.activations.sigmoid()
    dense = lambda W=W, act=leaky_relu : tf.keras.layers.Dense(W, activation=act, kernel_initializer='he_normal')

    inputs = tf.keras.Input(shape=(6 + 3*2*(rot_embed_dims_coarse) + 3*2*(pos_embed_dims_coarse))) #new (embedding dims (4) and (10) )
    outputs = inputs[:,:(3+3*2*(pos_embed_dims_coarse))] #only look at positional stuff for now
    for i in range(D):
        outputs = dense()(outputs)
        outputs = tf.keras.layers.LayerNormalization()(outputs) #old
        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs[:,:(3+3*2*(pos_embed_dims_coarse))]], -1)
            outputs = tf.keras.layers.LayerNormalization()(outputs) #as recomended by LOC-NDF 
    #combine to look at position and view direction together
    combined = tf.concat([outputs, inputs[:,(3+3*2*(pos_embed_dims_coarse)):]], -1)
    combined = dense(128, act=leaky_relu)(combined) #OG NeRF structure
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = dense(128, act=None)(combined)
    combined = tf.keras.layers.BatchNormalization()(combined)
    combined = dense(1, act=tf.keras.activations.sigmoid)(combined) #adding a 2nd channel for 1st vs 2nd return 
    model = tf.keras.Model(inputs=inputs, outputs=combined)
    
    return model



def cylindrical_to_cartesian(pts):

    x = pts[:,0]*tf.math.cos(pts[:,1])
    y = pts[:,0]*tf.math.sin(pts[:,1]) 
    z = pts[:,2]

    out = tf.transpose(tf.Variable([x, y, z]))

    return(out)

def add_patch(rays_o, rays_d, image):    
    """given tensors of rays origins (rays_o), rays directions (rays_d), and a training depth image, 
        return the corresponding point cloud in the world frame"""
    
    #OLD
    # #flatten first
    # rays_d_flat = np.reshape(rays_d, [-1,3])
    
    # #convert to spherical
    # rays_d_spherical = cartesian_to_spherical(rays_d_flat).numpy()

    # #reshape rays_d to same size as image to scale
    # rays_d_spherical = np.reshape(rays_d_spherical, [np.shape(rays_d)[0], np.shape(rays_d)[1], 3])
    # rays_d_spherical[:,:,0] *= image
    # rays_d_spherical = np.reshape(rays_d_spherical, [-1,3])
    # xyz = spherical_to_cartesian(rays_d_spherical)
    
    # xyz += np.reshape(rays_o, [-1,3])
    
    #NEW
    xyz = tf.reshape(rays_d * image[:,:,None], [-1,3]) + tf.reshape(rays_o, [-1,3])

    return xyz


def interpolate_missing_angles(pc1):
    """pc1 = cartesian coordinates of point cloud AFTER distortion correction has been applied"""

    pc1_spherical = cartesian_to_spherical(pc1)
    ray_drops = tf.where(pc1_spherical[:,0]<0.001)
    non_ray_drops = tf.where(pc1_spherical[:,0]>0.001)

    # Generate a regular 2D grid (source grid)
    source_grid_x, source_grid_y = np.meshgrid(np.linspace(0, 63, 64), np.linspace(0, 1023, 1024))
    source_points = np.column_stack((source_grid_x.flatten(), source_grid_y.flatten()))
    warped_points = pc1_spherical[:,1:].numpy()
#     print("warped_points", np.shape(warped_points))

    # Select known warped points (subset for interpolation)
    known_indices = non_ray_drops[:,0]
    known_source_points = source_points[known_indices]
    known_warped_points = warped_points[known_indices]

    # Interpolate missing points on the warped grid
    missing_indices = np.setdiff1d(np.arange(len(source_points)), known_indices)  # Remaining points
    missing_source_points = source_points[missing_indices]

    # Use griddata to estimate locations of missing points on the warped grid
    interpolated_points = griddata(known_source_points, known_warped_points, missing_source_points, method='cubic')
    # interpolated_points = np.nan_to_num(interpolated_points, 0)
#     print("\n interpolated_points", np.shape(interpolated_points), interpolated_points)

    #fill interpolated points back in to missing locations
    full_points_spherical = tf.zeros_like(pc1_spherical).numpy()[:,:2]
    #combine via mask old and new interpolated points
    full_points_spherical[non_ray_drops[:,0]] = known_warped_points
    full_points_spherical[ray_drops[:,0]] = interpolated_points

    full_points_spherical = np.append(np.ones([len(full_points_spherical), 1]), full_points_spherical, axis = 1)
    full_points = spherical_to_cartesian(full_points_spherical)
#     print("\n full_points_spherical", np.shape(full_points_spherical), tf.math.reduce_sum(full_points_spherical))

    return full_points

def get_rays_from_point_cloud(pc, m_hat, c2w):
    """pc = point cloud in cartesian coords
       m_hat = distortion correction states (in sensor frame)
       c2w = rigid transform from sensor to world frame"""

    # # better solution-- directly pass in undistorted point cloud and get view dirs from there...
    # # why didn't I do this sooner???
    # #    didn't do this before because we don't get a look direction for points with non-returns! 
    # #    I need to be clever about how to handle those situations!

    # #use bilinear interpolation to fill in view dirs where pixels are missing due to ray drop
    # dirs_undistorted = interpolate_missing_angles(pc)
    # dirs_undistorted = tf.cast(dirs_undistorted, tf.float32)

    # # #had this before
    # rotm = R.from_euler('xyz', [0,np.pi,np.pi]).as_matrix()
    # dirs_undistorted = dirs_undistorted @ rotm
    # dirs_undistorted = dirs_undistorted @ tf.cast(tf.linalg.pinv(c2w[:3, :3]), tf.float32)
    # # dirs_undistorted = dirs_undistorted @ tf.cast(c2w[:3, :3], tf.float32)

    # #Reshape directions
    # dirs = tf.reshape(dirs_undistorted, [1024, 64, 3])    #looks sharp
    # # dirs = tf.reverse(dirs, [1]) #flips right side up but causes more staggering
    # dirs = tf.transpose(dirs, [1,0,2]) #sharp

    # rays_d = tf.reduce_sum(dirs[..., tf.newaxis, :] * np.eye(3), -1)
    # rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))

    # print("rays_o", np.shape(rays_o))
    # print("rays_d", np.shape(rays_d))

    # return rays_o, rays_d

    ## Make view directions locus of points, distort, and use to reproject ~~~~~~~~~~~~~~~~~
    ## doesn't actually work with distortion, matches "redfix" demo in notebook (so at least gt transforms are right) 

    #make sure to update these if playing around with differnt patch sizes??
    H = 64
    W = 1024
    phimax_patch = np.deg2rad(-15.594) #worked best flipped (at least on old data pre-processing pipeline)
    phimin_patch = np.deg2rad(17.743)
    # phimin_patch = np.deg2rad(-15.594) #absolutely not!
    # phimax_patch = np.deg2rad(17.743)

    # get direction vectors of unit length for each point in cloud (rays_d)
    # need to be extra careful about non-returns
    # init completely full frustum of points as if sensor read 1m in every pixel for every direction
    i, j = tf.meshgrid(tf.range(1024, dtype=tf.float32), tf.range(64, dtype=tf.float32), indexing='xy')
    #[r, theta, phi]
    dirs_distorted = tf.stack([-tf.ones_like(i), #r
                          #theta
                        # -(i - ((W-1)/2))  /(W) * (2*np.pi/(1024//(W))) - np.pi/2, #-- had this before
                        (i - ((W-1)/2))  /(W) * (2*np.pi/(1024//(W))) - np.pi, #debugging in notebook
                          #phi
                        -((phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H-1))*(phimax_patch-phimin_patch)) -np.pi/2 #was this
                         ], -1)
    dirs_distorted = tf.transpose(dirs_distorted, [1,0,2]) #TEST
    dirs_distorted = tf.reshape(dirs_distorted,[-1,3])
    dirs_distorted = tf.reverse(dirs_distorted, [0]) #test
    dirs_distorted = spherical_to_cartesian(dirs_distorted)

    #apply distortion correction to that frustum as well
    # m_hat = np.array([3.,0.,0.,0.,0.,0.])#for debug    
    # print(m_hat)
    # dirs_undistorted = apply_motion_profile(dirs_distorted, m_hat, period_lidar=1.)
    dirs_undistorted = apply_motion_profile(dirs_distorted, 0.*m_hat, period_lidar=1.) #DEBUG-- why does this help???

    # #had this before
    # rotm = R.from_euler('xyz', [0,0,np.pi/2]).as_matrix()
    # dirs_undistorted = dirs_undistorted @ rotm
    # dirs_undistorted = dirs_undistorted @ tf.linalg.pinv(c2w[:3, :3])

    #debugging in notebook
    # rotm = R.from_euler('xyz', [0,0,np.pi]).as_matrix()
    # dirs_undistorted = dirs_undistorted @ rotm
    dirs_undistorted = dirs_undistorted @ tf.linalg.pinv(c2w[:3, :3])

    print("dirs_undistorted", np.shape(dirs_undistorted))

    # #TODO-- renormalize each element of dirs_unidistorted to be of unit length!
    # # print("\n dirs_undistorted:", np.shape(dirs_undistorted))
    # mag_dirs = tf.sqrt(tf.math.reduce_sum(dirs_undistorted**2, axis = 1))
    # # print("\n mag_dirs: \n", mag_dirs)
    # dirs_undistorted = dirs_undistorted / mag_dirs[:,None]

    # Reshape directions
    # dirs = tf.reshape(dirs_undistorted, [64, 1024, 3])    #old
    dirs = tf.reshape(dirs_undistorted, [1024, 64, 3])    #test
    dirs = tf.reverse(dirs, [1]) #test
    dirs = tf.transpose(dirs, [1,0,2]) #Test

    rays_d = tf.reduce_sum(dirs[..., tf.newaxis, :] * np.eye(3), -1)
    rays_o = tf.broadcast_to(c2w[:3, -1], tf.shape(rays_d))

    print("rays_o", np.shape(rays_o))
    print("rays_d", np.shape(rays_d))

    return rays_o, rays_d


def get_rays(H, W, c2w, phimin_patch, phimax_patch, debug = False):
    #IMPORTANT NOTE: this works for rendering but is insufficient for training from distorted point cloud data

    #when training on raw point cloud data, rays need to be determined from the coordinates of undistorted data!!!?!

    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')

    #Spherical projection model
    #[r, theta, phi]
    dirs_test = tf.stack([-tf.ones_like(i), #r
                          #theta
                          # (i - ((W-1)/2))  /(W) * (2*np.pi/(1024//(W))), #old (flipped horizontally)
                        # (-i + ((W-1)/2))  /(W) * (2*np.pi/(1024//(W))), #FIXES HORIZONTAL SKIPPING (had this pre 9/7)
                        # (-i + ((W)/2))  /(W-1) * (2*np.pi/(1024//(W))), #was this pre 8/23 -- seems to skip some horizontal scan lines at lower resolutions!!!
                        -(i - ((W-1)/2))  /(W) * (2*np.pi/(1024//(W))) - np.pi, #test 9/7, changing to match get_rays_from_pc
                          #phi
                          #need to manually account for elevation angle of patch 
                          #  (can not be inferred from c2w since that does not account for singularities near "poles" of spherical projection)
                        ((phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H-1))*(phimax_patch-phimin_patch)) -np.pi/2 #wa s this
                        # -((phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H))*(phimax_patch-phimin_patch)) -np.pi/2 #test
                         ], -1)
    dirs_test = tf.reshape(dirs_test,[-1,3])
    dirs_test = spherical_to_cartesian(dirs_test)

    dirs_test_OG = dirs_test.numpy().copy()
    
    # #old-- sensor rotates about about y in training data ~~~~~~~~~~~~~~~~~
    # rotm = R.from_euler('xyz', [np.pi/2,0,0]).as_matrix() #had this
    # dirs_test = dirs_test @ rotm

    # dirs_test = dirs_test @ tf.transpose(c2w[:3,:3])

    # rotm2 = R.from_euler('xyz', [np.pi/2,0,0]).as_matrix() #had this
    # dirs_test = dirs_test @ rotm2

    # dirs = dirs_test

    #new-- sensor rotates about about z in training data ~~~~~~~~~~~~~~~~~

    dirs_test = dirs_test @ tf.transpose(c2w[:3,:3])
    # dirs_test = dirs_test @ c2w[:3,:3]

    # rotm = R.from_euler('xyz', [np.pi,0,0]).as_matrix()
    # dirs_test = dirs_test @ rotm

    # rotm = R.from_euler('xyz', [0,0,np.pi]).as_matrix()
    # dirs_test = dirs_test @ rotm


    dirs = dirs_test

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    dirs = tf.reshape(dirs, [H,W,3])

    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * np.eye(3), -1)             
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))

    if debug == True:
        return rays_o, rays_d, dirs_test_OG
    else:
        return rays_o, rays_d

# #NEW -- updated integration strategy
# def render_rays(network_fn, rays_o, rays_d, z_vals):
    
#     def batchify(fn, chunk=1024*512): #1024*512 converged for v4 #1024*32 in TinyNeRF
#         return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

#     #Encode positions and directions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     ray_pos = rays_o[...,None,:] + rays_d[...,None,:] * z_vals
#     ray_pos_flat = tf.reshape(ray_pos, [-1, 3])
#     encoded_ray_pos = embed_fn(ray_pos_flat, pos_embed_dims) #10 embedding dims for pos
#     ray_dir = tf.reshape(rays_d[..., None,:]*tf.ones_like(z_vals, dtype = tf.float32), [-1,3]) #test
#     encoded_ray_dir = embed_fn(ray_dir, rot_embed_dims)  # embedding dims for dir
    
#     encoded_both = tf.concat([encoded_ray_pos, encoded_ray_dir], axis = -1)
#     raw = batchify(network_fn)(encoded_both) #old
#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#     raw = tf.reshape(raw, [ray_pos.shape[0],ray_pos.shape[1],-1,2]) #[sigma, ray_drop]

#     #CDF estimates output by network at each point
#     # CDF = tf.sigmoid(raw[...,0]) 
#     # CDF = tf.nn.tanh(raw[...,0]) 
#     CDF = tf.nn.relu(tf.nn.tanh(raw[...,0]))
#     ray_drop = raw[...,1] #ray drop chances


#     opacity = (CDF[:,:,1:]-CDF[:,:,:-1])/(1-CDF[:,:,:-1])
#     pad = tf.zeros_like(opacity[:,:,0])[:,:,None]
#     opacity = tf.concat([pad, opacity], axis = -1)
#     # print("\n raw: \n", raw[0,0,:10,0])
#     # print("\n CDF: ", tf.shape(CDF) ," \n", CDF[0,0,:10])
#     # print("opacity: \n", opacity[0,0,:10])

#     roll = tf.random.uniform(tf.shape(opacity))
#     hit_surfs = tf.argmax(roll < opacity, axis = -1)
#     # depth_map = tf.gather_nd(ray_pos, hit_surfs[:,:,None], batch_dims = 2) #nope
#     depth_map = tf.gather_nd(z_vals, hit_surfs[:,:,None], batch_dims = 2)[:,:,0] #nope
#     # rendered_points_flat = tf.reshape(rendered_points, [-1,3])

#     # print("depth_map:", tf.shape(depth_map  ))

#     # ray_drop_map = tf.reduce_sum(opacity * ray_drop, -1) #axis was -2, changed to -1 
#     #debug
#     temp = tf.reduce_mean(ray_drop, axis = -1) 
#     ray_drop_map = tf.ones_like(temp)

#     #old
#     # dists = tf.concat([z_vals[:,:,1:,:] - z_vals[:,:,:-1,:], tf.broadcast_to([1e10], z_vals[:,:,:1,:].shape)], -2) 
#     # dists = dists[:,:,:,0]
#     # alpha = 1.-tf.exp(-sigma_a * dists)
#     # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
#     # depth_map = tf.reduce_sum(weights * z_vals[:,:,:,0], -1)
#     # ray_drop_map = tf.reduce_sum(weights * ray_drop, -1) #axis was -2, changed to -1 


#     return depth_map, ray_drop_map, CDF


# adjusted from vanilla NeRF to pass in sample locations, still essentially just the image-nerf loss process
# @tf.function
def render_rays(network_fn, rays_o, rays_d, z_vals, fine = False, roll_override = None):
    """true batch size is determined by how many rays are in rays_o and rays_d"""

    # #DEBUG
    # rays_o = tf.cast(rays_o, tf.float32)
    # rays_d = tf.cast(rays_d, tf.float32)
    # z_vals = tf.cast(z_vals, tf.float32)

    #function for breaking down data to smaller pieces to hand off to the graphics card
    def batchify(fn, chunk=1024*512): #1024*512 converged for v4 #1024*32 in TinyNeRF
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    #Encode positions and directions 
    ray_pos = rays_o[...,None,:] + rays_d[...,None,:] * z_vals
    ray_pos_flat = tf.reshape(ray_pos, [-1, 3])
    encoded_ray_pos = embed_fn(ray_pos_flat, pos_embed_dims) #10 embedding dims for pos
    ray_dir = tf.reshape(rays_d[..., None,:]*tf.ones_like(z_vals, dtype = tf.float32), [-1,3]) #test
    encoded_ray_dir = embed_fn(ray_dir, rot_embed_dims)  # embedding dims for dir

    encoded_both = tf.concat([encoded_ray_pos, encoded_ray_dir], axis = -1)
    # print("encoded_both_fine", np.shape(encoded_both))

    raw = batchify(network_fn)(encoded_both)

    raw = tf.reshape(raw, [ray_pos.shape[0],ray_pos.shape[1],-1,2])

    # # #Image-NeRF volume rendering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    # sigma_a = tf.nn.relu(raw[...,0])
    # ray_drop = tf.nn.relu(raw[...,1])
    # dists = tf.concat([z_vals[:,:,1:,:] - z_vals[:,:,:-1,:], tf.broadcast_to([1e10], z_vals[:,:,:1,:].shape)], -2) 
    # dists = dists[:,:,:,0]
    # alpha = 1.-tf.exp(-sigma_a * dists)

    # # print("\n alpha", np.shape(alpha), alpha[0,0,:20])
    # CDF = 1-tf.math.cumprod((1-alpha), axis = -1)
    # # print("CDF: ", np.shape(CDF), CDF[0,0,:20])

    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # depth_map = tf.reduce_sum(weights * z_vals[:,:,:,0], -1)
    # ray_drop_map = tf.reduce_sum(weights * ray_drop, -1) #axis was -2, changed to -1 
    # acc_map = tf.reduce_sum(weights, -1)

    #Stochastic volume rendering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extract sigma_a and ray_drop predictions
    sigma_a = tf.nn.relu(raw[..., 0])
    # sigma_a = tf.sigmoid(raw[..., 0]) #test
    # ray_drop = tf.nn.relu(raw[..., 1])
    ray_drop = tf.sigmoid(raw[..., 1])

    ##TODO -- potential issue here???
    # Compute weights for volume rendering
    temp = z_vals[:,:,1:,0] - z_vals[:,:,:-1,0]
    # print("temp:", np.shape(temp))
    padding = tf.broadcast_to([100.], temp[:,:,0].shape)[:,:,None]
    # print("padding:", np.shape(padding))
    dists = tf.concat([temp, padding], axis=-1)

    # Convert sigma to detection probability
    alpha = 1. - tf.exp(-sigma_a * dists)
    # alpha = sigma_a * dists #TEST

    CDF = 1-tf.math.cumprod((1-alpha), axis = -1)
    # print("\n z_vals:", z_vals[0,0,:10,0])
    # print("dists: ", dists[0,0,:10])
    # print("\n sigma_a", np.shape(sigma_a), sigma_a[0,0,:]) #raw network output
    # print("alpha", np.shape(alpha), alpha[0,0,:]) #PDF
    # print("CDF: ", np.shape(CDF), CDF[0,0,:]) #CDF

    roll = tf.random.uniform(tf.shape(alpha))  #random sampling
    if roll_override is not None:
        roll = roll_override*tf.ones_like(alpha)
        # TEST --- look at first surfaces (first spike on CDF)
        # roll = 0.0001*tf.ones_like(alpha)
        # roll = 0.1*tf.ones_like(alpha)
        # roll = 0.2*tf.ones_like(alpha)
        # roll = 0.8*tf.ones_like(alpha)

    #TEST --- look at rear surfaces (last spike on CDF)
    # roll = 0.8*tf.ones_like(alpha)

    # hit_surfs = tf.argmax(roll < alpha, axis = -1)
    # depth_map = tf.gather_nd(z_vals, hit_surfs[:,:,None], batch_dims = 2)[:,:,0]
    # # weights = alpha * tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True) #was this
    # weights = np.gradient(CDF, axis = 2) + 1e-8 #works but fuzzy
    # # weights = tf.cast(roll < alpha, tf.float32) * alpha * tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True) #test
    # # print("alpha", tf.shape(alpha))
    # # print("hit_surfs", tf.shape(hit_surfs))

    # # Compute ray_drop_map using the same weights
    # ray_drop_map = tf.reduce_sum(weights * ray_drop, axis=-1) #works ish (but not great)
    # # ray_drop_map = tf.reduce_max(ray_drop, axis=-1) #should be better but was failing
    # acc_map = tf.reduce_sum(weights, axis=-1)

    # #first in line-of-sight (direct depth) rendering ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sigma_a = tf.nn.relu(raw[...,0]) #[0, 10+]
    ray_drop = tf.nn.relu(raw[...,1])
    alpha = 1. - tf.exp(-sigma_a * z_vals[:,:,:,0])
    
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    depth_map = tf.reduce_sum(weights * z_vals[:,:,:,0], -1)

    ray_drop_map = tf.reduce_sum(weights * ray_drop, -1) #axis was -2, changed to -1 
    acc_map = tf.reduce_sum(weights, -1)

    
    # return depth_map, acc_map, ray_drop_map, weights #old strategy
    return depth_map, ray_drop_map, CDF, weights

def calculate_loss_Mai_City(depth, ray_drop, target, target_drop_mask, 
    d1 = None, d2 = None, CDF = None, gtCDF = None):
    """special case of loss calculation for Mai City Dataset (which has no ray drop points)"""

    #ray drop loss
    L_raydrop = tf.keras.losses.binary_crossentropy(target_drop_mask, ray_drop)
    L_raydrop = tf.math.reduce_mean(tf.abs(L_raydrop))

    #masked distance loss (suppressing ray drop areas)
    depth_nondrop = tf.math.multiply(depth, target_drop_mask)
    target_nondrop = tf.math.multiply(target, target_drop_mask)
    L_dist = tf.reduce_sum(tf.abs(depth_nondrop - target_nondrop))#old
    # L_dist = tf.reduce_sum((depth_nondrop - target_nondrop)**2) #new 
    # print("L_dist old", tf.shape(L_dist)) 

    # #Try Huber Loss instead of simple masked distance loss
    # depth_nondrop = tf.math.multiply(depth, target_drop_mask)
    # target_nondrop = tf.math.multiply(target, target_drop_mask)
    # abs_error = tf.abs(depth_nondrop - target_nondrop)
    # delta = 0.0025 #0.05 
    # quadratic = tf.minimum(abs_error, delta)
    # linear = abs_error - quadratic
    # L_dist = tf.reduce_mean(0.5 * quadratic**2 + delta * linear)
    # # print("L_dist new", tf.shape(L_dist))
    
    #Gradient Loss (structural regularization for smooth surfaces) -- (LiDAR-NeRF method) ~~~~~~~~~~~
#     thresh = 0.025 #was at 0.025, set to 0.1 in LiDAR-NeRF
    ##--seems like this works better if I set different values for each component
    #    this makes sense since the resolution of the sensor is differnt in horizontal and vertical(?)
    thresh_horiz = 0.05 #for Newer College 
    thresh_vert = 0.005 #for Newer College
    # thresh_horiz = 10. #turns it off 
    # thresh_vert = 10. #turns it off
    mask = np.ones(np.shape(target[:,:,0]))
    vertical_grad_target = np.gradient(target[:,:,0])[0] 
    vertical_past_thresh = np.argwhere(tf.abs(vertical_grad_target) > thresh_vert) #old
    # #test for double gradient 
    # vertical_grad_target2 = np.gradient(vertical_grad_target)[0] 
    # vertical_past_thresh = np.argwhere(tf.abs(vertical_grad_target2) > thresh_vert)

    mask[vertical_past_thresh[:,0], vertical_past_thresh[:,1]] = 0 #1
    horizontal_grad_target = np.gradient(target[:,:,0])[1]
    horizontal_past_thresh = np.argwhere(tf.abs(horizontal_grad_target) > thresh_horiz) #old
    # #test for double gradient
    # horizontal_grad_target2 = np.gradient(horizontal_grad_target)[1]  
    # horizontal_past_thresh = np.argwhere(tf.abs(horizontal_grad_target2) > thresh_horiz)
    mask[horizontal_past_thresh[:,0], horizontal_past_thresh[:,1]] = 0 #1
    
    vertical_grad_inference = np.gradient(depth[:,:,0])[0]
    horizontal_grad_inference = np.gradient(depth[:,:,0])[1]
    # mag_difference = tf.math.sqrt((vertical_grad_target-vertical_grad_inference)**2 + (horizontal_grad_target-horizontal_grad_inference)**2)
    #DEBUG -- use struct reg. to amplify LR in sharp corners 
    mag_difference = tf.reduce_mean(tf.abs(depth_nondrop - target_nondrop)) 

    #suppress ray drop areas (for distance and gradient loss)
    L_reg = np.multiply(mag_difference, mask)
    L_reg = L_reg[:,:,None]
    L_reg = tf.reduce_mean(tf.math.multiply(L_reg, target_drop_mask))
    L_reg = tf.cast(L_reg, tf.float32)         

    lam0 = 10 #0 
    lam1 = 100 #100 
    lam2 = 1000. #1 
    lam4 = 0.1

    # #if we're using CDF for loss instead of distance error ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # if CDF is not None:

    #     #suppress regular distance loss
    #     lam0 = 0.

    #     CDFdiff = tf.abs(CDF - gtCDF)
    #     # CDFdiff = tf.math.multiply(CDFdiff, target_drop_mask)

    #    # # ~~~ prevent gradient mask from getting rid of double returns in windows, etc.
    #    #  save_non_ground = tf.zeros_like(mask).numpy()
    #    #  #NEED TO TURN OFF WHEN WE HAVE MULTIPLE VERTICAL PATCHES 
    #    #  save_non_ground[:40,:] = 1 #prevent anything in the top ~3/4 of image from getting masked
    #    #  save_non_groud = tf.convert_to_tensor(save_non_ground)
    #    #  together = tf.concat([save_non_groud[:,:,None], mask[:,:,None]], axis = -1)
    #    #  mask = tf.math.reduce_max(together, axis = -1)
    #    #  mask = tf.cast(mask, tf.float32)
    #    #  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #     # mask = tf.cast(mask, tf.float32)

    #     # suppress high gradient regions to minimize bias induced by beam spreading
    #     # CDFdiff_low_grad = tf.math.multiply(CDFdiff, mask[:,:,None])
    #     # CDFdiff = 0.2*CDFdiff + 0.8*CDFdiff_low_grad
 
    #     # print("L1:", tf.math.reduce_sum(CDFdiff))
    #     # print("L2:", tf.math.reduce_sum(CDFdiff**2))

    #     # CDF_loss = tf.reduce_sum(CDFdiff) #L1 Loss
    #     # CDF_loss = tf.reduce_sum(CDFdiff**2) #L2 Loss
    #     CDF_loss = tf.reduce_sum(CDFdiff**2 + CDFdiff) #using both (was this) -- works much better!
    #     # CDF_loss = tf.reduce_sum(0.1*(CDFdiff**2) + 0.9*CDFdiff) #test 8/8-- weight L1 more heavily

    #     # loss = lam0*L_dist + lam1*L_reg + lam2*L_raydrop + lam4*CDF_loss #had this for newer college
    #     loss = lam4*CDF_loss #don't worry about raydrop on Mai City
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    loss = lam0*L_dist + lam1*L_reg + lam2*L_raydrop#LiDAR-NeRF

    # loss = lam0*L_dist + lam1*L_reg + lam2*L_raydrop + lam4*CDF_loss
    # print("\n L_dist: ", lam0*L_dist, "\n L_raydrop:", lam2*L_raydrop, "\n L_CDF:", lam4*CDF_loss)

    return(loss)


def calculate_loss(depth, ray_drop, target, target_drop_mask, 
    d1 = None, d2 = None, CDF = None, gtCDF = None):
    """L_total = L_dist + lam1*L_intensity + lam2*L_raydrop + lam3*L_reg"""

    #ray drop loss
    L_raydrop = tf.keras.losses.binary_crossentropy(target_drop_mask, ray_drop)
    L_raydrop = tf.math.reduce_mean(tf.abs(L_raydrop))

    #masked distance loss (suppressing ray drop areas)
    depth_nondrop = tf.math.multiply(depth, target_drop_mask)
    target_nondrop = tf.math.multiply(target, target_drop_mask)
    L_dist = tf.reduce_mean(tf.abs(depth_nondrop - target_nondrop))
    # print("L_dist old", tf.shape(L_dist))

    # #Try Huber Loss instead of simple masked distance loss
    # depth_nondrop = tf.math.multiply(depth, target_drop_mask)
    # target_nondrop = tf.math.multiply(target, target_drop_mask)
    # abs_error = tf.abs(depth_nondrop - target_nondrop)
    # delta = 0.0025 #0.05 
    # quadratic = tf.minimum(abs_error, delta)
    # linear = abs_error - quadratic
    # L_dist = tf.reduce_mean(0.5 * quadratic**2 + delta * linear)
    # # print("L_dist new", tf.shape(L_dist))
    
    #Gradient Loss (structural regularization for smooth surfaces) -- (LiDAR-NeRF method) ~~~~~~~~~~~
#     thresh = 0.025 #was at 0.025, set to 0.1 in LiDAR-NeRF
    ##--seems like this works better if I set different values for each component
    #    this makes sense since the resolution of the sensor is differnt in horizontal and vertical(?)
    # thresh_horiz = 0.05 #for Newer College 
    # thresh_vert = 0.005 #for Newer College
    thresh_horiz = 10. #turns it off 
    thresh_vert = 10. #turns it off
    mask = np.ones(np.shape(target[:,:,0]))
    vertical_grad_target = np.gradient(target[:,:,0])[0] 
    vertical_past_thresh = np.argwhere(tf.abs(vertical_grad_target) > thresh_vert) #old
    # #test for double gradient 
    # vertical_grad_target2 = np.gradient(vertical_grad_target)[0] 
    # vertical_past_thresh = np.argwhere(tf.abs(vertical_grad_target2) > thresh_vert)

    mask[vertical_past_thresh[:,0], vertical_past_thresh[:,1]] = 0 #1
    horizontal_grad_target = np.gradient(target[:,:,0])[1]
    horizontal_past_thresh = np.argwhere(tf.abs(horizontal_grad_target) > thresh_horiz) #old
    # #test for double gradient
    # horizontal_grad_target2 = np.gradient(horizontal_grad_target)[1]  
    # horizontal_past_thresh = np.argwhere(tf.abs(horizontal_grad_target2) > thresh_horiz)
    mask[horizontal_past_thresh[:,0], horizontal_past_thresh[:,1]] = 0 #1
    
    vertical_grad_inference = np.gradient(depth[:,:,0])[0]
    horizontal_grad_inference = np.gradient(depth[:,:,0])[1]
    # mag_difference = tf.math.sqrt((vertical_grad_target-vertical_grad_inference)**2 + (horizontal_grad_target-horizontal_grad_inference)**2)
    #DEBUG -- use struct reg. to amplify LR in sharp corners 
    mag_difference = tf.reduce_mean(tf.abs(depth_nondrop - target_nondrop)) 

    #suppress ray drop areas (for distance and gradient loss)
    L_reg = np.multiply(mag_difference, mask)
    L_reg = L_reg[:,:,None]
    L_reg = tf.reduce_mean(tf.math.multiply(L_reg, target_drop_mask))
    L_reg = tf.cast(L_reg, tf.float32)         

    #if we're using CDF for loss instead of distance error ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if CDF is not None:
        CDFdiff = tf.abs(CDF - gtCDF)
        CDFdiff = tf.math.multiply(CDFdiff, target_drop_mask)

       # # ~~~ prevent gradient mask from getting rid of double returns in windows, etc.
       #  save_non_ground = tf.zeros_like(mask).numpy()
       #  #NEED TO TURN OFF WHEN WE HAVE MULTIPLE VERTICAL PATCHES 
       #  save_non_ground[:40,:] = 1 #prevent anything in the top ~3/4 of image from getting masked
       #  save_non_groud = tf.convert_to_tensor(save_non_ground)
       #  together = tf.concat([save_non_groud[:,:,None], mask[:,:,None]], axis = -1)
       #  mask = tf.math.reduce_max(together, axis = -1)
       #  mask = tf.cast(mask, tf.float32)
       #  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        mask = tf.cast(mask, tf.float32)


        #TEST-- suppress high gradient regions to minimize bias induced by beam spreading
        CDFdiff_low_grad = tf.math.multiply(CDFdiff, mask[:,:,None])
        CDFdiff = 0.2*CDFdiff + 0.8*CDFdiff_low_grad
 
        # print("L1:", tf.math.reduce_sum(CDFdiff))
        # print("L2:", tf.math.reduce_sum(CDFdiff**2))

        # CDF_loss = tf.reduce_sum(CDFdiff) #L1 Loss
        # CDF_loss = tf.reduce_sum(CDFdiff**2) #L2 Loss
        CDF_loss = tf.reduce_sum(CDFdiff**2 + CDFdiff) #using both (was this) -- works much better!
        # CDF_loss = tf.reduce_sum(0.1*(CDFdiff**2) + 0.9*CDFdiff) #test 8/8-- weight L1 more heavily

        # print("CDF_loss", CDF_loss)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    lam0 = 0. #10 #NEED TO USE THIS WHEN SCALING DOWN EVERYTHING TO FIT IN UNIT BOX(?)
              # othersize loss gets dominated by raydrop?? 
    lam1 = 0. #100 
    lam2 = 1000. #1 
    lam4 = 0.1

    # print("\n lam2*L_raydrop", lam2*L_raydrop)
    # print("lam4*CDF_loss", lam4*CDF_loss)

    # loss = lam0*L_dist + lam1*L_reg + lam2*L_raydrop #LiDAR-NeRF

    loss = lam2*L_raydrop + lam4*CDF_loss
    # print("\n L_dist: ", lam0*L_dist, "\n L_raydrop:", lam2*L_raydrop, "\n L_CDF:", lam4*CDF_loss)

    return(loss)

def apply_motion_profile(cloud_xyz, m_hat, period_lidar = 1):
    """Linear correction for motion distortion, using ugly python code

    cloud_xyz: distorted cloud in Cartesian space
    m_hat: estimated motion profile for linear correction
    period_lidar: time it takes for LIDAR sensor to record a single sweep
    """

    period_base = (2*np.pi)/m_hat[-1]

    #remove inf values
    cloud_xyz = cloud_xyz[cloud_xyz[:,0] < 10_000]
    #convert to spherical coordinates
    # cloud_spherical = c2s(cloud_xyz).numpy()

    #Because of body frame yaw rotation, we're not always doing a full roation - we need to "uncurl" initial point cloud
    # (this is NOT baked in to motion profile)
    # cloud_spherical = cloud_spherical[np.argsort(cloud_spherical[:,1])] #sort by azim angle
    #get total overlap in rotation between LIDAR and base frames (since both are rotating w.r.t. world Z)
    # point of intersection = (t_intersection) * (angular velocity base)
    #                       = ((n * T_a * T_b) / (T_a + T_b)) * omega_base 
    # total_rot = -2*np.pi*np.sin(m_hat[-1]/(-m_hat[-1] + (2*np.pi/period_lidar)))
    # print("total_rot:", total_rot)

    #scale linearly starting at theta = 0
    # cloud_spherical[:,1] = ((cloud_spherical[:,1]) % (2*np.pi))*((2*np.pi - total_rot)/(2*np.pi)) + total_rot #works

    #sort by azim angle again- some points will have moved past origin in the "uncurling" process
    # cloud_spherical = cloud_spherical[np.argsort(cloud_spherical[:,1])] 

    #reorient
    # cloud_spherical[:,1] = ((cloud_spherical[:,1] + np.pi) % (2*np.pi)) - np.pi
    # cloud_xyz = s2c(cloud_spherical).numpy() #convert back to xyz

    rectified_vel  = -m_hat[None,:]
    # rectified_vel[0,-1] = 0 #zero out yaw since we already compensated for it

    T = (2*np.pi)/(-m_hat[-1] + (2*np.pi/period_lidar)) #time to complete 1 scan #was this
    # print(T)
    rectified_vel = rectified_vel * T #was this

    # # using yaw angles ~~~~~~~~~~~~~~~~~~~~~~    
    yaw_angs = cartesian_to_spherical(cloud_xyz)[:,1].numpy() #standard -- (used in VICET paper)
    # yaw_angs = -cartesian_to_spherical(cloud_xyz)[:,1].numpy() #flipped


    last_subzero_idx = int(len(yaw_angs) // 8)
    yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] = yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] + 2*np.pi

    #jump in <yaw_angs> is causing unintended behavior in real world LIDAR data
    yaw_angs = (yaw_angs + 2*np.pi)%(2*np.pi)

    motion_profile = (yaw_angs / np.max(yaw_angs))[:,None] @ rectified_vel #was this
    # motion_profile = yaw_angs[:,None] @ rectified_vel #test
    # print("\n new: \n", motion_profile[:,0])
    # self.yaw_angs_new = yaw_angs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #Apply motion profile
    x = -motion_profile[:,0]
    y = -motion_profile[:,1]
    z = -motion_profile[:,2]
    phi = motion_profile[:,3]
    theta = motion_profile[:,4]
    psi = motion_profile[:,5]

    T_rect_numpy = np.array([[cos(psi)*cos(theta), 
        sin(psi)*cos(theta), 
        -sin(theta), 
        x*cos(psi)*cos(theta) + y*sin(psi)*cos(theta) - z*sin(theta)], 
        [sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi), sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi), 
        sin(phi)*cos(theta), x*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi)) + y*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)) + z*sin(phi)*cos(theta)],
         [sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi), -sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi), cos(phi)*cos(theta), 
         x*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi)) - y*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi)) + z*cos(phi)*cos(theta)], 
         [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x))]])
    T_rect_numpy = np.transpose(T_rect_numpy, (2,0,1))
    cloud_homo = np.append(cloud_xyz, np.ones([len(cloud_xyz),1]), axis = 1)

    undistorted_pc =  (T_rect_numpy @ cloud_homo[:,:,None]).astype(np.float32)

    return undistorted_pc[:,:3,0]


def spherical_to_cartesian(pts):
    """converts spherical -> cartesian"""

    x = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.cos(pts[:,1])
    y = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.sin(pts[:,1]) 
    z = pts[:,0]*tf.math.cos(pts[:,2])

    out = tf.transpose(tf.Variable([x, y, z]))
    # out = tf.Variable([x, y, z])
    return(out)

def cylindrical_to_cartesian(pts):

    x = pts[:,0]*tf.math.cos(pts[:,1]) #old
    y = pts[:,0]*tf.math.sin(pts[:,1]) #old
    z = pts[:,2]

    out = tf.transpose(tf.Variable([x, y, z]))

    return(out)

def cartesian_to_spherical(pts):
    """ converts points from cartesian coordinates to spherical coordinates """
    r = tf.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
    phi = tf.math.acos(pts[:,2]/r)
    theta = tf.math.atan2(pts[:,1], pts[:,0])

    out = tf.transpose(tf.Variable([r, theta, phi]))
    return(out)

def crop_floaters(nerf_pc, threshold = 0.1):
    
    # Create an Open3D point cloud from the NumPy array
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nerf_pc)

    # Compute the nearest neighbors
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    points_to_keep = []
    for i in range(len(pcd.points)):
        # Perform a nearest neighbor search for the current point (excluding itself)
        [k, idx, distances] = kdtree.search_knn_vector_3d(pcd.points[i], 2)

        if k > 1:
            # Check the distance to the nearest neighbor
            nearest_distance = np.sqrt(distances[1])

            if nearest_distance <= threshold:
                points_to_keep.append(i)

    # Select points that meet the condition
    filtered_pcd = pcd.select_by_index(points_to_keep)
    # print(type(filtered_pcd))

    nerf_pc = np.asarray(filtered_pcd.points)
    
    return(nerf_pc)

# def crop_floaters(nerf_pc):
    
#     # Create an Open3D point cloud from the NumPy array
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(nerf_pc)

#     # Define the base threshold and scaling factor
#     base_threshold = 0.0  # Base threshold (adjust as needed)
#     scaling_factor = 0.005  # Determines how the threshold grows with distance

#     # Compute the nearest neighbors
#     kdtree = o3d.geometry.KDTreeFlann(pcd)

#     points_to_keep = []
#     for i in range(len(pcd.points)):
#         point = np.asarray(pcd.points[i])

#         # Compute the distance of the point from the origin
#         distance_from_origin = np.linalg.norm(point)

#         # Scale the threshold based on the point's distance from the origin
#         scaled_threshold = base_threshold + scaling_factor * distance_from_origin

#         # Perform a nearest neighbor search for the current point (excluding itself)
#         [k, idx, distances] = kdtree.search_knn_vector_3d(pcd.points[i], 2)

#         if k > 1:
#             # Check the distance to the nearest neighbor
#             nearest_distance = np.sqrt(distances[1])

#             # Keep the point if the nearest neighbor distance is within the scaled threshold
#             if nearest_distance <= scaled_threshold:
#                 points_to_keep.append(i)
# #             else:
# #                 print("throwing out")

#     # Select points that meet the condition
#     filtered_pcd = pcd.select_by_index(points_to_keep)

#     nerf_pc = np.asarray(filtered_pcd.points)
    
#     return(nerf_pc)
