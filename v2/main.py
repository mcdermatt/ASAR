import numpy as np
import tensorflow as tf
#need to have these two lines to work on my ancient 1060 3gb
#  https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from utils import *
import tensorflow_probability as tfp
import time
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
from ICET3D import ICET3D

"""Runs the same script as the ICET3D notebook but with better performance due to not using inline display"""

#NOTE: Out of Memory Error comes from too high fidelity/ pts in cloud tensor --> 100x100x2x120,000 > 2gb


plt = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)

basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'
frame_range = range(150, 151, 1)
dataset = pykitti.raw(basedir, date, drive)
velo1 = dataset.get_velo(0) # Each scan is a Nx4 array of [x,y,z,reflectance]
cloud1 = velo1[:,:3]
cloud1_tensor = tf.convert_to_tensor(cloud1, np.float32)
velo2 = dataset.get_velo(2) # Each scan is a Nx4 array of [x,y,z,reflectance]
cloud2 = velo2[:,:3]
cloud2_tensor = tf.convert_to_tensor(cloud2, np.float32)

nc = 5
mnp = 15
npts = 100000
D = True #draw sim
DG = False #draw grid
DE = True #draw ellipsoids
DC = True #draw correspondences
TD = True #use test dataset
CM = "voxel" #correspondence method, "voxel" or "NN"

start = time.time()


# #use whole point set
# #---------------------------------------------------------------------------------
# f = tf.constant([50,50,2]) #fidelity in x, y, z # < 5s
# lim = tf.constant([-100.,100.,-100.,100.,-10.,10.]) #needs to encompass every point
# Q, x_hist = ICET3D(cloud1_tensor[:npts], cloud2_tensor[:npts], plt, bounds = lim, 
#            fid = f, num_cycles = nc , min_num_pts = mnp, draw = D, draw_grid = DG, 
#            draw_ell = DE, draw_corr = DC)

# #---------------------------------------------------------------------------------

#just consider small section of image where there are easily identifiable features:
#----------------------------------------------------------------------------------
limtest = tf.constant([-20.,0.,-20.,0.,-1.5,1.5])
f = tf.constant([10,10,10])
# cloud1_tensor = tf.squeeze(tf.gather(cloud1_tensor, tf.where( (cloud1_tensor[:,0] > limtest[0]))))	#only works one cond at a time
cloud1_tensor = tf.squeeze(tf.gather(cloud1_tensor, tf.where( tf.math.reduce_all(tf.concat( (
	(cloud1_tensor[:,0] > limtest[0])[:,None], 
	(cloud1_tensor[:,0] < limtest[1])[:,None], 
	(cloud1_tensor[:,1] > limtest[2])[:,None], 
	(cloud1_tensor[:,1] < limtest[3])[:,None],
	(cloud1_tensor[:,2] > limtest[4])[:,None], 
	(cloud1_tensor[:,2] < limtest[5])[:,None],
	), axis = 1 ), axis = 1))))

cloud2_tensor = tf.squeeze(tf.gather(cloud2_tensor, tf.where( tf.math.reduce_all(tf.concat( (
	(cloud2_tensor[:,0] > limtest[0])[:,None], 
	(cloud2_tensor[:,0] < limtest[1])[:,None], 
	(cloud2_tensor[:,1] > limtest[2])[:,None], 
	(cloud2_tensor[:,1] < limtest[3])[:,None],
	(cloud2_tensor[:,2] > limtest[4])[:,None], 
	(cloud2_tensor[:,2] < limtest[5])[:,None],), axis = 1 ), axis = 1))))

Q, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = limtest, 
           fid = f, num_cycles = nc , min_num_pts = mnp, draw = D, draw_grid = DG,
           draw_ell = DE, draw_corr = DC, test_dataset = TD, CM = CM)

#----------------------------------------------------------------------------------


print("took", time.time() - start, "seconds total")