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

from utils import R_tf
from metpy.calc import lat_lon_grid_deltas
from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt
import copy
import trimesh


from pillow_heif import register_heif_opener
from matplotlib import pyplot as p
from colmapParsingUtils import *
from scipy.spatial.transform import Rotation as R
import cv2

from tqdm import tqdm_notebook as tqdm
from PIL import Image

tf.compat.v1.enable_eager_execution()


def posenc(x, embed_dims):
  rets = [x]
  for i in range(embed_dims):
    for fn in [tf.sin, tf.cos]:
      rets.append(fn(2.**i * x))
  return tf.concat(rets, -1)

#2**18 is waaaaay too fine to learn anything on most of the channels!?!
# L_embed =  5 #18 #15 #10 #6
pos_embed_dims = 14 #14 #15
embed_fn = posenc

def init_model(D=8, W=256): #8,256
#     relu = tf.keras.layers.ReLU() #OG NeRF   
    relu = tf.keras.layers.LeakyReLU() #per LOC-NDF   
    dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act, kernel_initializer='glorot_uniform')

#     inputs = tf.keras.Input(shape=(3 + 3*2*L_embed)) #old (embed everything together)
    inputs = tf.keras.Input(shape=(6 + 3*2*(4) + 3*2*(pos_embed_dims))) #new (embedding dims (4) and (10) )
    # outputs = inputs #old
    outputs = inputs[:,:(3+3*2*(pos_embed_dims))] #only look at positional stuff for now

    #try removing view dependant effects from density 

    for i in range(D):
        outputs = dense()(outputs)
        outputs = tf.keras.layers.LayerNormalization()(outputs) #as recomended by LOC-NDF 
        
        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs[:,:(3+3*2*(pos_embed_dims))]], -1)

    #extend small MLP after output of density channel to get ray drop
    sigma_channel = dense(1, act=None)(outputs)    
    rd_start = tf.concat([outputs, inputs[:,(3+3*2*(pos_embed_dims)):]], -1)
    rd_channel = dense(256, act=relu)(outputs) #OG NeRF structure
    # rd_channel = dense(256, act=relu)(outputs) #test adding another channel
    rd_channel = dense(128, act=relu)(rd_channel)
    rd_channel = dense(1, act=tf.keras.activations.sigmoid)(rd_channel)
    out = tf.concat([sigma_channel, rd_channel], -1)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    
    return model

def cylindrical_to_cartesian(pts):

    x = pts[:,0]*tf.math.cos(pts[:,1])
    y = pts[:,0]*tf.math.sin(pts[:,1]) 
    z = pts[:,2]

    out = tf.transpose(tf.Variable([x, y, z]))

    return(out)


def get_rays(H, W, c2w, phimin_patch, phimax_patch):
    i, j = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')

    #~~~~~~~~~~~~~~~~~~~~~~~~~
    # #Cylindrical projection model (new) -- working better 5/27
    # dirs_test = tf.stack([-tf.ones_like(i), #r
    #                   #theta
    #                   # (i - (1024//(2*n_rots)))  /(2048//(2*n_rots)) * (2*np.pi/n_rots) + np.pi, #for uninterpolated images
    #                   (i - (W//2))  /(W) * (2*np.pi/(1024//W)) + np.pi, #just use W
    #                   #phi
    #                   # (phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H-1))*(phimax_patch-phimin_patch) #-np.pi/2 #using 5/1
    #                  np.arcsin((phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H-1))*(phimax_patch-phimin_patch)) #-np.pi/2 #TEST
    #                  ], -1)
    # dirs_test = tf.reshape(dirs_test,[-1,3])
    # dirs_test = cylindrical_to_cartesian(dirs_test)
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~
    #Spherical projection model (old)
    #[r, theta, phi]
    dirs_test = tf.stack([-tf.ones_like(i), #r
                          #theta
                            # (i - (W//2))  /(W) * (2*np.pi/(1024//W)), #old-- unequal split!
                          # (i - ((W-1)/2))  /(W-1) * (2*np.pi/(1024//(W-1))), #test-- equal split
                          (i - ((W-1)/2))  /(W) * (2*np.pi/(1024//(W))), #test-- slightly more accurate
                          #phi
                          #need to manually account for elevation angle of patch 
                          #  (can not be inferred from c2w since that does not account for singularities near "poles" of spherical projection)
                          # (phimax_patch + phimin_patch)/2 + ((-j+(H/2))/(H))*(phimax_patch-phimin_patch) -np.pi/2 #slightly wrong
                        # (phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H-1))*(phimax_patch-phimin_patch) -np.pi/2 #using 5/1
                        -((phimax_patch + phimin_patch)/2 - ((-j+((H-1)/2))/(H-1))*(phimax_patch-phimin_patch)) -np.pi/2 #Seems to fix issue with z axis translation flip!
                         ], -1)
    dirs_test = tf.reshape(dirs_test,[-1,3])
    dirs_test = spherical_to_cartesian(dirs_test)

    #TEST
    # dirs_test[:,3] = -dirs_test[:,3]

    #~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # rotm = R.from_euler('xyz', [0,-np.pi/2 + (phimax + phimin)/2,0]).as_matrix() #was this --> think I'm double counting horizon centering
    rotm = R.from_euler('xyz', [0,-np.pi/2,0]).as_matrix() #was this
    dirs_test = dirs_test @ rotm
    dirs_test = dirs_test @ tf.transpose(c2w[:3,:3])
    dirs = dirs_test @ (c2w[:3,:3] 
                          @ R.from_euler('xyz', [0,0,np.pi/2]).as_matrix()
                          @ np.linalg.pinv(c2w[:3,:3]) )

    dirs = tf.reshape(dirs, [H,W,3])

    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * np.eye(3), -1)     
        
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))
    return rays_o, rays_d

#new method (pass in sample locations)
def render_rays(network_fn, rays_o, rays_d, z_vals):
    
    def batchify(fn, chunk=1024*512): #1024*512 converged for v4 #1024*32 in TinyNeRF
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

#     #Old version (pass in pts directly, no seperation of positions and directions) ~~~~~~
#     #[image_height, image_width, batch_size, 3]
#     pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals
#     # Run network to estimate densities and ray drop 
#     pts_flat = tf.reshape(pts, [-1,3])
#     pts_flat = embed_fn(pts_flat)
#     raw = batchify(network_fn)(pts_flat)
#     #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #New version-- encode both positions and directions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ray_pos = rays_o[...,None,:] + rays_d[...,None,:] * z_vals
    ray_pos_flat = tf.reshape(ray_pos, [-1, 3])
    encoded_ray_pos = embed_fn(ray_pos_flat, pos_embed_dims) #10 embedding dims for pos
    ray_dir = tf.reshape(rays_d[..., None,:]*tf.ones_like(z_vals, dtype = tf.float32), [-1,3]) #test
    encoded_ray_dir = embed_fn(ray_dir, 4)  # embedding dims for dir

#     print("ray_pos", np.shape(ray_pos))
#     print("ray_dir", np.shape(ray_dir))
#     print("encoded_ray_pos", np.shape(encoded_ray_pos))
#     print("encoded_ray_dir", np.shape(encoded_ray_dir))
    
    encoded_both = tf.concat([encoded_ray_pos, encoded_ray_dir], axis = -1)
#     print("encoded_both", np.shape(encoded_both))
    raw = batchify(network_fn)(encoded_both) #old

    # print("raw", np.shape(raw))
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # print("problem here: ", [ray_pos.shape[0],ray_pos.shape[1],-1,2])    
#     raw = tf.reshape(raw, list(pts.shape[:-1]) + [2]) # [depth, ray drop] #old

    raw = tf.reshape(raw, [ray_pos.shape[0],ray_pos.shape[1],-1,2]) # test -- doesn't allow upscaling when rendering?
#     print("raw", np.shape(raw))
    
    sigma_a = tf.nn.relu(raw[...,0])
    ray_drop = tf.nn.relu(raw[...,1])
    
    # Do volume rendering with unique z vals for each ray
    dists = tf.concat([z_vals[:,:,1:,:] - z_vals[:,:,:-1,:], tf.broadcast_to([1e10], z_vals[:,:,:1,:].shape)], -2) 
    dists = dists[:,:,:,0]

    # print("sigma_a", np.shape(sigma_a))
    # print("dists", np.shape(dists))
    
    alpha = 1.-tf.exp(-sigma_a * dists)  
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    depth_map = tf.reduce_sum(weights * z_vals[:,:,:,0], -1)
    ray_drop_map = tf.reduce_sum(weights * ray_drop, -1) #axis was -2, changed to -1 
    acc_map = tf.reduce_sum(weights, -1)
    
    return depth_map, acc_map, ray_drop_map, weights

def calculate_loss(depth, ray_drop, target, target_drop_mask):
    """L_total = L_dist + lam1*L_intensity + lam2*L_raydrop + lam3*L_reg"""

    #ray drop loss
    L_raydrop = tf.keras.losses.binary_crossentropy(target_drop_mask, ray_drop)
    L_raydrop = tf.math.reduce_mean(tf.abs(L_raydrop))

    #distance loss (suppressing ray drop areas)
    depth_nondrop = tf.math.multiply(depth, target_drop_mask)
    target_nondrop = tf.math.multiply(target, target_drop_mask)
    L_dist = tf.reduce_mean(tf.abs(depth_nondrop - target_nondrop))
    
    #Gradient Loss (structural regularization for smooth surfaces) -- (LiDAR-NeRF method) ~~~~~~~~~~~
#     thresh = 0.025 #was at 0.025, set to 0.1 in LiDAR-NeRF
    ##--seems like this works better if I set different values for each component
    #    this makes sense since the resolution of the sensor is differnt in horizontal and vertical(?)
    thresh_horiz = 0.025 
    thresh_vert = 0.025
    # thresh_horiz = 0.001 #test 
    # thresh_vert = 0.01 #test
    mask = np.ones(np.shape(target[:,:,0]))
    vertical_grad_target = np.gradient(target[:,:,0])[0] 
    # vertical_past_thresh = np.argwhere(tf.abs(vertical_grad_target) > thresh_vert) #old
    # #test for double gradient 
    vertical_grad_target2 = np.gradient(vertical_grad_target)[0] 
    vertical_past_thresh = np.argwhere(tf.abs(vertical_grad_target2) > thresh_vert)

    mask[vertical_past_thresh[:,0], vertical_past_thresh[:,1]] = 0 #1
    horizontal_grad_target = np.gradient(target[:,:,0])[1]
    # horizontal_past_thresh = np.argwhere(tf.abs(horizontal_grad_target) > thresh_horiz) #old
    # #test for double gradient
    horizontal_grad_target2 = np.gradient(horizontal_grad_target)[1]  
    horizontal_past_thresh = np.argwhere(tf.abs(horizontal_grad_target2) > thresh_horiz)
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

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        

    lam1 = 0 #100 #100 #10 #100
    lam2 = 1 #1/(64**2)
    loss = L_dist + lam1*L_reg + lam2*L_raydrop       

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
    # print(rectified_vel[:,-1])
    # rectified_vel[:-1] = rectified_vel[:-1] * T #nope
    # rectified_vel[:,-1] = rectified_vel[:,-1] * T #also nope

    # linearly spaced motion profile ~~~~~~~~~~~~~~~~~~~~~
    # this is a bad way of doing it ... what happens if most of the points are on one half of the scene??

    # part2 = np.linspace(0.5, 1.0, len(cloud_xyz)//2)[:,None]
    # part1 = np.linspace(0, 0.5, len(cloud_xyz) - len(cloud_xyz)//2)[:,None]
    # motion_profile = np.append(part1, part2, axis = 0) @ rectified_vel

    # # using yaw angles ~~~~~~~~~~~~~~~~~~~~~~
    #  (NEW)
    
    #TODO: need to center point cloud before getting yaw angles

    yaw_angs = cartesian_to_spherical(cloud_xyz)[:,1].numpy()
    last_subzero_idx = int(len(yaw_angs) // 8)
    yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] = yaw_angs[last_subzero_idx:][yaw_angs[last_subzero_idx:] < 0.3] + 2*np.pi
    # yaw_angs = yaw_angs / (2*np.pi) #test
    # yaw_angs = yaw_angs[:,None]  #test

    # #test
    # self.yaw_angs = yaw_angs
    # self.cloud_xyz = cloud_xyz
    # color = 255*self.yaw_angs/(2*np.pi)
    # cname = np.array([255-color, color, 255-color]).T.tolist()
    # # self.disp.append(Points(self.cloud_xyz + 0.01*np.random.randn(np.shape(cloud_xyz)[0],3), c=cname, r = 3))
    # self.disp.append(Points(self.cloud_xyz, c=cname, r = 3))

    #jump in <yaw_angs> is causing unintended behavior in real world LIDAR data
    yaw_angs = (yaw_angs + 2*np.pi)%(2*np.pi)

    #TODO: should I use (2pi - T) in place of max(yaw_angs) -> ???
    motion_profile = (yaw_angs / np.max(yaw_angs))[:,None] @ rectified_vel #was this
    # motion_profile = yaw_angs[:,None] @ rectified_vel #test
    # print("\n new: \n", motion_profile[:,0])
    # self.yaw_angs_new = yaw_angs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    #Apply motion profile
    # # Old loopy method ~~~~~~~~~~~~~~
    # T = []
    # for i in range(len(motion_profile)):
    #   tx, ty, tz, roll, pitch, yaw = motion_profile[i]
    #   R = np.dot(np.dot(np.array([[1, 0, 0], 
    #                               [0, np.cos(roll), -np.sin(roll)], 
    #                               [0, np.sin(roll), np.cos(roll)]]), 
    #                   np.array([[np.cos(pitch), 0, np.sin(pitch)], 
    #                             [0, 1, 0], 
    #                             [-np.sin(pitch), 0, np.cos(pitch)]])), 
    #                   np.array([[np.cos(yaw), -np.sin(yaw), 0], 
    #                             [np.sin(yaw), np.cos(yaw), 0], 
    #                             [0, 0, 1]]))
    #   T.append(np.concatenate((np.concatenate((R, np.array([[tx], [ty], [tz]])), axis=1), np.array([[0, 0, 0, 1]])), axis=0))
    
    # #should be the same size:
    # # print(len(T))
    # # Apply inverse of motion transformation to each point
    # undistorted_pc = np.zeros_like(cloud_xyz)
    # for i in range(len(cloud_xyz)):
    #   point = np.concatenate((cloud_xyz[i], np.array([1])))
    #   T_inv = np.linalg.inv(T[i])
    #   corrected_point = np.dot(T_inv, point)[:3]
    #   undistorted_pc[i] = corrected_point
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # new method ~~~~~~~~~~~~~~~~~~~~~~

    x = -motion_profile[:,0]
    y = -motion_profile[:,1]
    z = -motion_profile[:,2]
    phi = motion_profile[:,3]
    theta = motion_profile[:,4]
    psi = motion_profile[:,5]

    #need to inverse this
    # T_rect_numpy = np.array([[cos(psi)*cos(theta), sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi), sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi), -x], [sin(psi)*cos(theta), sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi), -sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi), -y], [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta), -z], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x))]])
    #to this
    T_rect_numpy = np.array([[cos(psi)*cos(theta), sin(psi)*cos(theta), -sin(theta), x*cos(psi)*cos(theta) + y*sin(psi)*cos(theta) - z*sin(theta)], [sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi), sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi), sin(phi)*cos(theta), x*(sin(phi)*sin(theta)*cos(psi) - sin(psi)*cos(phi)) + y*(sin(phi)*sin(psi)*sin(theta) + cos(phi)*cos(psi)) + z*sin(phi)*cos(theta)], [sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi), -sin(phi)*cos(psi) + sin(psi)*sin(theta)*cos(phi), cos(phi)*cos(theta), x*(sin(phi)*sin(psi) + sin(theta)*cos(phi)*cos(psi)) - y*(sin(phi)*cos(psi) - sin(psi)*sin(theta)*cos(phi)) + z*cos(phi)*cos(theta)], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.ones(len(x))]])
    T_rect_numpy = np.transpose(T_rect_numpy, (2,0,1))
    # print(np.shape(cloud_xyz))
    cloud_homo = np.append(cloud_xyz, np.ones([len(cloud_xyz),1]), axis = 1)
    # print("cloud homo", np.shape(cloud_homo))

    # undistorted_pc =  (np.linalg.pinv(T_rect_numpy) @ cloud_homo[:,:,None]).astype(np.float32)
    undistorted_pc =  (T_rect_numpy @ cloud_homo[:,:,None]).astype(np.float32)
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
