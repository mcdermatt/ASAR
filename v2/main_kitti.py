import numpy as np
import tensorflow as tf

#need to have these two lines to work on my ancient 1060 3gb
#  https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#-----uncomment to force run on CPU----------
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# if tf.test.gpu_device_name():
#     print('GPU found')
# else:
#     print("No GPU found")
#--------------------------------------------

from utils import *
import tensorflow_probability as tfp
import time
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
from ICET3D import ICET3D

"""Runs ICET on a SINGLE PAIR of scans from the KITTI dataset"""

nc1 = 1	 #number of cycles using coarse voxel sizes
mnp = 3 #50 #minimum number of points per voxel
D = True #draw sim
DG = False #draw grid
DE = True #draw ellipsoids
DC = False #draw correspondences
TD = False #use test dataset
CM = "voxel" #correspondence method, "voxel" or "NN"
vizL = False #draw arrows in direction of non-truncated directions for each distribution
id1 = 55 #idx of 1st scan #118 is the naughty scan in raw 0005, 37 is bad in benchmark 05
id2 = id1 + 1 #idx of 2nd scan

plt = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)
# plt = Plotter(N=1, axes=4, interactive=True)

# ## Use raw data ----------------------------------------------------------------
basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005' #urban
# drive = '0009' #suburban #len = 446
# drive = '0018' #highway traffic
dataset = pykitti.raw(basedir, date, drive)

# #-------------------------------------------------------------------------------

## Use benchmark data ----------------------------------------------------------
# basepath = "E:/KITTI/dataset/"
# # sequence = '05'
# sequence = '00'

# dataset = pykitti.odometry(basepath, sequence)
#-------------------------------------------------------------------------------


velo1 = dataset.get_velo(id1) # Each scan is a Nx4 array of [x,y,z,reflectance]
cloud1 = velo1[:,:3]
#remove far away points from cloud1
limy = 30 #horizotal lim in +/- directions
lim2 = -1.65 #vertical lim (sensor is +1.73m from ground)
# cloud1 = cloud1[ cloud1[:,0] < lim1]
# cloud1 = cloud1[ cloud1[:,0] > -lim1]
# cloud1 = cloud1[ cloud1[:,1] < limy]
# cloud1 = cloud1[ cloud1[:,1] > -limy]
# cloud1 = cloud1[1:100000:1000,:]
# cloud1 = np.array([0.3,0.3,0.1])*np.random.randn(100,3)
cloud1 = cloud1[ cloud1[:,2] > lim2]
cloud1_tensor = tf.convert_to_tensor(cloud1, np.float32)

velo2 = dataset.get_velo(id2) # Each scan is a Nx4 array of [x,y,z,reflectance]
cloud2 = velo2[:,:3]
#repeat for cloud2
# cloud2 = cloud2[ cloud2[:,0] < lim1]
# cloud2 = cloud2[ cloud2[:,0] > -lim1]
# cloud2 = cloud2[ cloud2[:,1] < limy]
# cloud2 = cloud2[ cloud2[:,1] > -limy]
# cloud2 = cloud2[1:100000:1000,:]
# cloud2 = np.array([0.3,0.3,0.1])*np.random.randn(100,3)
cloud2 = cloud2[ cloud2[:,2] > lim2]
cloud2_tensor = tf.convert_to_tensor(cloud2, np.float32)

# ---------------------------------------------------------------------------------

# #use whole point set
# #---------------------------------------------------------------------------------
# f = tf.constant([50,50,5]) #wa s 50,50,2... #fidelity in x, y, z # < 5s --- works for 005
# f = tf.constant([20,20,2]) #need larger voxel sizes for 018
f = tf.constant([100,100,4]) #test
lim = tf.constant([-100.,100.,-100.,100.,-10.,10.]) #needs to encompass every point
# lim = tf.constant([-100.,100.,-50.,50.,-5.,5.])
# npts = 100000
Q, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = lim, 
           fid = f, num_cycles = nc1 , min_num_pts = mnp, draw = D, draw_grid = DG, 
           draw_ell = DE, draw_corr = DC, test_dataset = TD, CM = CM, vizL = vizL)

# f2 = tf.constant([40,40,4])
# Q2, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = lim, 
# 			fid = f2, num_cycles = 5, draw_ell = False, draw_corr = DC, 
# 			min_num_pts = 20, draw = D, vizL = vizL, xHat0 = x_hist[-1])
# #---------------------------------------------------------------------------------

#just consider small section of image where there are easily identifiable features:
#----------------------------------------------------------------------------------
# limtest = tf.constant([-20.,0.,-20.,0.,-1.5,1.5])
# # f = tf.constant([35,35,35])
# f = tf.constant([21,21,4])
# # f = tf.constant([17,17,17])

# cloud1_tensor = tf.squeeze(tf.gather(cloud1_tensor, tf.where( (cloud1_tensor[:,0] > limtest[0]))))	#only works one cond at a time
# cloud1_tensor = tf.squeeze(tf.gather(cloud1_tensor, tf.where( tf.math.reduce_all(tf.concat( (
# 	(cloud1_tensor[:,0] > limtest[0])[:,None], 
# 	(cloud1_tensor[:,0] < limtest[1])[:,None], 
# 	(cloud1_tensor[:,1] > limtest[2])[:,None], 
# 	(cloud1_tensor[:,1] < limtest[3])[:,None],
# 	(cloud1_tensor[:,2] > limtest[4])[:,None], 
# 	(cloud1_tensor[:,2] < limtest[5])[:,None],
# 	), axis = 1 ), axis = 1))))

# cloud2_tensor = tf.squeeze(tf.gather(cloud2_tensor, tf.where( tf.math.reduce_all(tf.concat( (
# 	(cloud2_tensor[:,0] > limtest[0])[:,None], 
# 	(cloud2_tensor[:,0] < limtest[1])[:,None], 
# 	(cloud2_tensor[:,1] > limtest[2])[:,None], 
# 	(cloud2_tensor[:,1] < limtest[3])[:,None],
# 	(cloud2_tensor[:,2] > limtest[4])[:,None], 
# 	(cloud2_tensor[:,2] < limtest[5])[:,None],), axis = 1 ), axis = 1))))

# Q, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = limtest, 
#            fid = f, num_cycles = nc , min_num_pts = mnp, draw = D, draw_grid = DG,
#            draw_ell = DE, draw_corr = DC, test_dataset = TD, CM = CM, vizL = vizL)

#----------------------------------------------------------------------------------

ans = np.sqrt(abs(Q.numpy())) 
print("\n predicted solution standard deviation of error: \n", ans[0,0], ans[1,1], ans[2,2], ans[3,3], ans[4,4], ans[5,5])

