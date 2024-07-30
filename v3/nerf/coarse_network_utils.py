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
from lidar_nerf_utils import *

tf.compat.v1.enable_eager_execution()

def run_coarse_network(model_coarse, z_vals_coarse, rays_o, rays_d,n_resample = 128):
#     print(np.shape(z_vals_coarse))
#     print(np.shape(rays_o))
#     print(np.shape(rays_d))

    
    def batchify(fn, chunk=1024*512):
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    
    #encode positions and directions
    ray_pos = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_coarse
    ray_pos_flat = tf.reshape(ray_pos, [-1, 3])
    encoded_ray_pos = embed_fn(ray_pos_flat, pos_embed_dims) #10 embedding dims for pos
    ray_dir = tf.reshape(rays_d[..., None,:]*tf.ones_like(z_vals_coarse, dtype = tf.float32), [-1,3]) #test
    encoded_ray_dir = embed_fn(ray_dir, rot_embed_dims)  # embedding dims for dir
    encoded_both = tf.concat([encoded_ray_pos, encoded_ray_dir], axis = -1)
    
    #pass to network
    # ouput size [H, W, 1]
    weights_coarse = batchify(model_coarse)(encoded_both)
    weights_coarse = tf.reshape(weights_coarse, [ray_pos.shape[0],ray_pos.shape[1],-1])
    
    #calculate widths of each bin
    width_coarse = tf.experimental.numpy.diff(z_vals_coarse[:,:,:,0], axis = 2)
    padding_config = [[0, 0],[0, 0],[0, 1]]
    width_coarse = tf.pad(width_coarse, padding_config, constant_values=0.001)

#     print(np.shape(z_vals_coarse))
#     print(np.shape(weights_coarse))
#     print(np.shape(width_coarse))
    
    
    #resample according to histogram output by coarse proposal network
    z_vals_fine = resample_z_vals(z_vals_coarse, weights_coarse[:,:,:,None], width_coarse[:,:,:,None], n_resample=n_resample)
    
    return z_vals_fine, weights_coarse


#updated to run in parallel about two batch dimensions
def resample_z_vals(z_vals_coarse, weights_coarse, w_coarse, n_resample=128):
    """use inverse transform sampling to pick new z vals for fine network about where coarse network density
     is highest"""
    
    # print("\n z_vals_coarse \n", np.shape(z_vals_coarse))
#     print("\n weights_coarse \n",np.shape(weights_coarse))
#     print("\n width_coarse \n",np.shape(w_coarse))
    
    zc = z_vals_coarse[:,:,:,0]
    wc = weights_coarse[:,:,:,0]
    width_coarse = w_coarse[:,:,:,0]

    # Pad weights for CDF computation, removing the first padded value after
    wc_padded = tf.pad(wc, [[0, 0], [0, 0], [1, 0]], constant_values=0)[:, :, :-1]

    # Compute the cumulative sum (CDF) across the last axis
    wc_cdf = tf.math.cumsum(wc_padded / tf.math.reduce_sum(wc_padded, axis=-1, keepdims=True), axis=-1)

    # Generate uniform random samples, sorting to ensure they're in CDF order
    randy = tf.sort(tf.random.uniform([z_vals_coarse.shape[0], z_vals_coarse.shape[1], n_resample]), axis=-1)

    # Find the indices in the CDF where the random samples should be inserted
    idx = tf.searchsorted(wc_cdf, randy, side='right')

    # Gather CDF and z-values for left and right indices
    cdf_left = tf.gather(wc_cdf, idx - 1, batch_dims=2)
    cdf_right = tf.gather(wc_cdf, idx, batch_dims=2)
    values_left = tf.gather(zc, idx - 1, batch_dims=2)
    values_right = tf.gather(zc, idx, batch_dims=2)

    # Interpolate to get the continuous sample values
    weights = (randy - cdf_left) / (cdf_right - cdf_left)
    z_vals_new = values_left + weights * (values_right - values_left)

    return z_vals_new

# continuous_samples = resample_z_vals(z_vals_coarse, weights_coarse, w_coarse, H = H, W=W)


def calculate_loss_coarse_network(z_vals_coarse, z_vals_fine, weights_coarse, weights_fine):
    '''Calculate loss for coarse network. Given histograms for scene density outptut by fine network,
    see how close the density estimated by the coarse network got us.'''
    
    print("\n calculate loss coarse network \n")
    
    def safe_segment_sum(data, segment_ids, num_segments):
        # Initialize an array to store the segment sums
        segment_sums = tf.zeros((num_segments,), dtype=data.dtype)
        # Compute the segment sums using tensor_scatter_nd_add
        segment_sums = tf.tensor_scatter_nd_add(segment_sums, tf.expand_dims(segment_ids, 1), data)
        return segment_sums


    def run_on_ray(index): 
        i, j = index
        zc = z_vals_coarse[i, j]
        zf = z_vals_fine[i, j]
        wc = weights_coarse[i, j]
        wf = weights_fine[i, j]
        width_fine_for_ray = width_fine[i,j] 
        width_coarse_for_ray = width_coarse[i,j]
        
        idx = tf.searchsorted(zc, zf, side='right') - 1
        fine_sum = safe_segment_sum(wf*width_fine_for_ray, idx, len(zc))/width_coarse_for_ray        

        mask = tf.cast(fine_sum > wc , tf.float32)
        L_i = tf.math.reduce_sum(mask*(fine_sum-wc))
        return L_i
    
    #get spacing between subsequent z measurements in z_vals_fine to weight contributions
    width_fine = tf.experimental.numpy.diff(z_vals_fine, axis = 2)
    padding_config = [[0, 0],[0, 0],[0, 1]]
    width_fine = tf.pad(width_fine, padding_config, constant_values=0.001)
    
    width_coarse = tf.experimental.numpy.diff(z_vals_coarse, axis = 2)
    padding_config = [[0, 0],[0, 0],[0, 1]]
    width_coarse = tf.pad(width_coarse, padding_config, constant_values=0.001)

    #normalize coarse and fine to contain an area of 1
    area_coarse = tf.math.reduce_sum(weights_coarse * width_coarse, axis =2)[:,:,None]
    weights_coarse = weights_coarse/area_coarse
    area_fine = tf.math.reduce_sum(weights_fine * width_fine, axis =2)[:,:,None]
    weights_fine = weights_fine/area_fine
    # print("sum weights coarse:", tf.math.reduce_sum(weights_coarse * width_coarse[:,:,:], axis = 2)) #will be all ones if norm'd correctly
    # print("sum weights fine:",tf.math.reduce_sum(weights_fine * width_fine[:,:,:], axis = 2)) #will be all ones if norm'd correctly

    #map over each batch dimension
    indices = tf.stack(tf.meshgrid(tf.range(z_vals_coarse.shape[0]), tf.range(z_vals_coarse.shape[1]), indexing='ij'), axis=-1)
    indices = tf.reshape(indices, (-1, 2))
    
    # Use tf.map_fn to apply the run_on_ray() in parallel over batch dimensions    
    L = tf.map_fn(run_on_ray, indices, fn_output_signature=tf.float32)
    L = tf.reshape(L, (z_vals_coarse.shape[0], z_vals_coarse.shape[1]))
    
    return L

# L = calculate_loss_coarse_network(z_vals_coarse[...,0], z_vals_fine[...,0], weights_coarse[...,0], weights_fine[...,0])