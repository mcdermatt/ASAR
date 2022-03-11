from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET

# ## init KITTI dataset -----------------------------------------------------------------
# basedir = 'C:/kitti/'
# date = '2011_09_26'
# drive = '0005'
# idx = 10
# frame_range = range(150, 151, 1)
# dataset = pykitti.raw(basedir, date, drive)
# velo1 = dataset.get_velo(idx) # Each scan is a Nx4 array of [x,y,z,reflectance]
# c1 = velo1[:,:3]
# # c1 = c1[c1[:,2] > -1.5] #ignore ground plane
# velo2 = dataset.get_velo(idx+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
# c2 = velo2[:,:3]
# # c2 = c2[c2[:,2] > -1.5] #ignore ground plane
# ## ------------------------------------------------------------------------------------


## load custom point cloud geneated in matlab------------------------------------------
c1 = np.loadtxt("scene1_scan1.txt", dtype = float)
c2 = np.loadtxt("scene1_scan2.txt", dtype = float)
# c1 = c1[c1[:,2] > -1.55] #ignore ground plane
# c2 = c2[c2[:,2] > -1.55] #ignore ground plane
## ------------------------------------------------------------------------------------

# #single distinct cluster---------------------------------------------------------------
# c1 = np.random.randn(3000,3)*tf.constant([0.3,0.04,0.3]) + tf.constant([0.,4.,0.])
# c2 = np.random.randn(3000,3)*tf.constant([0.3,0.04,0.3]) + tf.constant([0.,4.,0.]) - np.array([0.1, 0.1, 0.0])
# # c2 = c1 - np.array([0.1, 0.3, 0.0])
# # -------------------------------------------------------------------------------------

it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 4, draw = True)
# it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 5, draw = True, x0 = it.X)

# print("\n predicted solution error covariance: \n", it.pred_stds)

ViewInteractiveWidget(it.plt.window)