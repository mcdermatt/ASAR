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

pos_embed_dims_coarse = 18#8 #18
rot_embed_dims_coarse = 6#3 #6

def run_coarse_network(model_coarse, z_vals_coarse, width_coarse, rays_o, rays_d,n_resample = 128):
    
    def batchify(fn, chunk=1024*512):
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    
    #encode positions and directions
    ray_pos = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_coarse
    ray_pos_flat = tf.reshape(ray_pos, [-1, 3])
    encoded_ray_pos = embed_fn(ray_pos_flat, pos_embed_dims_coarse) #10 embedding dims for pos
    ray_dir = tf.reshape(rays_d[..., None,:]*tf.ones_like(z_vals_coarse, dtype = tf.float32), [-1,3]) #test
    encoded_ray_dir = embed_fn(ray_dir, rot_embed_dims_coarse)  # embedding dims for dir
    encoded_both = tf.concat([encoded_ray_pos, encoded_ray_dir], axis = -1)
    
    #pass to network
    # ouput size [H, W, 1]
    weights_coarse = batchify(model_coarse)(encoded_both)
    weights_coarse = tf.reshape(weights_coarse, [ray_pos.shape[0],ray_pos.shape[1],-1])

    width_coarse = width_coarse[:,:,:,0]    
    # print(z_vals_coarse[0,0,:,0])
    # print(width_coarse[0,0,:])

    # print(np.shape(z_vals_coarse))
    # print(np.shape(weights_coarse))
    # print(np.shape(width_coarse))    

    #rescale add small uniform probibility of selection for each bin
    eps = 1e-3 #1e-3
    weights_coarse_scaled = weights_coarse + eps*tf.ones_like(weights_coarse)
    weights_coarse_scaled = weights_coarse_scaled/ tf.math.reduce_sum(width_coarse*weights_coarse_scaled, axis = 2)[:,:,None]
    # print("should all be 1: \n", tf.math.reduce_sum(weights_coarse * width_coarse, axis = 2))
    # print(np.shape(weights_coarse))

    #resample according to histogram output by coarse proposal network
    z_vals_fine, width_fine = resample_z_vals(z_vals_coarse - width_coarse[:,:,:,None]/2, weights_coarse_scaled[:,:,:,None], width_coarse[:,:,:,None], n_resample=n_resample)
    # z_vals_fine, width_fine = resample_z_vals(z_vals_coarse, weights_coarse_scaled[:,:,:,None], width_coarse[:,:,:,None], n_resample=n_resample)

    return z_vals_fine, width_fine, weights_coarse_scaled
    # return z_vals_fine, width_fine, weights_coarse #return raw output of network, not scaled to 1

#updated to run in parallel about two batch dimensions
def resample_z_vals(z_vals_coarse, weights_coarse, w_coarse, n_resample=128):
    zc = z_vals_coarse[:, :, :, 0]
    wc = weights_coarse[:, :, :, 0]
    width_coarse = w_coarse[:, :, :, 0]

    #TODO: issue is here??
    # Pad weights for CDF computation, removing the first padded value after
    wc_padded = tf.pad(wc, [[0, 0], [0, 0], [1, 0]], constant_values=0)[:, :, :-1] #old
    # wc_padded = tf.pad(wc, [[0, 0], [0, 0], [0, 1]], constant_values=0)[:, :, :-1] #debug - nope
    # wc_padded = wc #test
    # print("wc_padded", wc_padded[0,0,:])
    # print("wc", wc[0,0,:])

    # if np.sum(np.isnan(weights_coarse[:,:,:,0])) > 0:
    #     print("actually found a NaN value")
    #     print(wc[0,0,:])

    wc_padded = tf.where(tf.math.is_nan(wc_padded), 0.001*tf.ones_like(wc_padded), wc_padded) #test

    tf.debugging.check_numerics(wc_padded, "NaN found in wc_padded inside resample_z_vals")
    
    # print("wc_padded max", tf.math.reduce_max(wc_padded))
    # print("wc_padded min", tf.math.reduce_min(wc_padded))

    # Compute the sum of padded weights and ensure it's non-zero
    sum_wc_padded = tf.math.reduce_sum(wc_padded, axis=-1, keepdims=True)
    tf.debugging.check_numerics(sum_wc_padded, "NaN found in sum_wc_padded inside resample_z_vals")

    # Avoid division by zero by setting a small value where the sum is zero
    epsilon = 1e-6
    # sum_wc_padded = tf.where(sum_wc_padded == 0, epsilon, sum_wc_padded)
    sum_wc_padded = tf.where(sum_wc_padded < epsilon, epsilon, sum_wc_padded)

    # Compute the cumulative sum (CDF) across the last axis
    wc_cdf = tf.math.cumsum(wc_padded / sum_wc_padded, axis=-1)
    tf.debugging.check_numerics(wc_cdf, "NaN found in wc_cdf inside resample_z_vals")

    # print("wc_cdf max", tf.math.reduce_max(wc_cdf))
    # print("wc_cdf min", tf.math.reduce_min(wc_cdf))
    # print("weights sum", tf.math.reduce_sum(weights_coarse, axis = 2))

    # Generate uniform random samples, sorting to ensure they're in CDF order
    #entirely random
    # randy = tf.sort(tf.random.uniform([z_vals_coarse.shape[0], z_vals_coarse.shape[1], n_resample]), axis=-1)
    # #repeating coarse bin locations (makes calculating loss easier)
    randy = tf.sort(tf.random.uniform([z_vals_coarse.shape[0], z_vals_coarse.shape[1], n_resample - z_vals_coarse.shape[2]]), axis=-1)

    # Find the indices in the CDF where the random samples should be inserted
    idx = tf.searchsorted(wc_cdf, randy, side='right')

    # Clip indices to ensure they are within bounds
    idx = tf.clip_by_value(idx, 1, tf.shape(wc_cdf)[-1] - 1)

    # Gather CDF and z-values for left and right indices
    cdf_left = tf.gather(wc_cdf, idx - 1, batch_dims=2)
    cdf_right = tf.gather(wc_cdf, idx, batch_dims=2)
    values_left = tf.gather(zc, idx - 1, batch_dims=2)
    values_right = tf.gather(zc, idx, batch_dims=2)

    # Add epsilon to avoid division by zero during interpolation
    denom = cdf_right - cdf_left + epsilon

    # Check for zero denominators explicitly and set to epsilon
    denom = tf.where(denom == 0, epsilon, denom)


    # Interpolate to get sample values at the LEFT edge of each histogram bin
    weights = (randy - cdf_left) / denom
    z_vals_new = values_left + weights * (values_right - values_left)
    z_vals_new = z_vals_new - z_vals_new[:,:,:1] #make sure first bin starts at 0


    ## if repeating coarse bin locations  ~~~~~~~~~~~~~~~~
    z_vals_new = tf.concat([z_vals_new, z_vals_coarse[:,:,:,0]], axis = 2)
    z_vals_new = tf.sort(z_vals_new, axis = 2)
    # print(np.shape(z_vals_new))
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    #get the centers of each histogram bin
    width_new = tf.experimental.numpy.diff(z_vals_new, axis=2)
    width_new = tf.concat([width_new, 1.- z_vals_new[:,:,-1][:,:,None] ], axis=2)
    z_vals_new = z_vals_new + width_new/2

    return z_vals_new, width_new


def safe_segment_sum(data, segment_ids, num_segments):
    # Extract dimensions
    batch_size, num_rays, num_samples = tf.shape(data)
    
    # Flatten data and segment_ids
    data_flat = tf.reshape(data, [-1])
    segment_ids_flat = tf.reshape(segment_ids, [-1])
    
    # Initialize segment_sums tensor
    segment_sums = tf.zeros([batch_size * num_rays, num_segments], dtype=data.dtype)
    
    # Calculate indices for tensor_scatter_nd_add
    indices = tf.stack([
        tf.repeat(tf.range(batch_size * num_rays), num_samples),  # Repeat for each sample
        segment_ids_flat
    ], axis=1)
    
    # Scatter the data
    segment_sums = tf.tensor_scatter_nd_add(segment_sums, indices, data_flat)
    
    # Reshape segment_sums to match original shape
    segment_sums = tf.reshape(segment_sums, [batch_size, num_rays, num_segments])
    
    return segment_sums

def calculate_loss_coarse_network(z_vals_coarse, z_vals_fine, weights_coarse, weights_fine, 
                                    width_coarse,width_fine, debug = False):
    '''Calculate loss for coarse network. Given histograms for scene density output by fine network,
    see how close the density estimated by the coarse network got us.'''

    #move bins to start at 0
    # z_vals_fine = z_vals_fine - width_fine / 2

    # Normalize coarse and fine weights
    area_coarse = tf.reduce_sum(weights_coarse * width_coarse, axis=2, keepdims=True)
    weights_coarse /= area_coarse
    area_fine = tf.reduce_sum(weights_fine * width_fine, axis=2, keepdims=True)
    weights_fine /= area_fine

    # print("\n test \n", z_vals_coarse[0,0,:] - width_coarse[0,0,:]/2)

    # Compute the index for gathering width_coarse
    # idx = tf.searchsorted(z_vals_coarse, z_vals_fine, side='right') - 1 #old
    idx = tf.searchsorted(z_vals_coarse - width_coarse/2, z_vals_fine, side='right') - 1 #center in bin?
    # idx = tf.searchsorted(z_vals_coarse - width_coarse/2, z_vals_fine, side='right') #test offset by 1 bin??

    idx = tf.clip_by_value(idx, 0, z_vals_coarse.shape[2] - 1)
    fine_sum = safe_segment_sum(weights_fine * width_fine, idx, z_vals_coarse.shape[2])
    fine_sum /= width_coarse
    # print("fine_sum \n", fine_sum[0,0,:])

    # Calculate the final loss
    mask = tf.cast(fine_sum > weights_coarse, tf.float32)
    L = tf.reduce_sum(mask * (fine_sum - weights_coarse) * width_coarse, axis=2) #scale by width of each coarse ray

    # print(tf.math.reduce_sum(weights_fine * width_fine, axis = 2)) #will be all ones if norm'd correctly
    # print(tf.math.reduce_sum(weights_coarse * width_coarse, axis = 2)) #will be all ones if norm'd correctly
    if debug:
        return L, fine_sum
    else:
        return L
