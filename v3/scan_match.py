from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf

#problem at 114??

# # init KITTI dataset -----------------------------------------------------------------
# basedir = 'C:/kitti/'
# date = '2011_09_26'

# # urban dataset used in 3D-ICET paper 
# # drive = '0005'
# # idx = 117

# #test with aiodrive
# drive = 'aiodrive'
# idx = 1

# #alternate dataset with fewer moving objects?
# # drive = '0009'
# # idx = 245
# # drive = '0093'
# # idx = 220

# dataset = pykitti.raw(basedir, date, drive)
# velo1 = dataset.get_velo(idx) # Each scan is a Nx4 array of [x,y,z,reflectance]
# c1 = velo1[:,:3]
# velo2 = dataset.get_velo(idx+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
# c2 = velo2[:,:3]
# # c1 = c1[c1[:,2] > -1.5] #ignore ground plane
# # c2 = c2[c2[:,2] > -1.5] #ignore ground plane
# # c1 = c1[c1[:,2] > -2.] #ignore reflections
# # c2 = c2[c2[:,2] > -2.] #ignore reflections

# #load previously processed cloud 1
# # c1 = np.loadtxt("cloud1_good.txt")
# # ------------------------------------------------------------------------------------

#TIERS forest dataset -----------------------------------------------------------------

filename1 = 'C:/TIERS/rawPointClouds/scan0.txt'
filename2 = 'C:/TIERS/rawPointClouds/scan1.txt'

c1 = np.loadtxt(filename1, dtype = float)
c2 = np.loadtxt(filename2, dtype = float)

# #---------------------------------------------------------------------------------------

# #CODD (colaborative driving dataset)----------------------------------------------------
# import h5py

# # filename = 'C:/CODD/data/m1v7p7s769.hdf5' #straight line urban(?)
# # filename = 'C:/CODD/data/m5v10p6s31.hdf5' #turn on country road
# filename = 'C:/CODD/data/m2v7p3s333.hdf5'


# vidx = 0 #vehicle index
# idx = 2 #frame idx


# with h5py.File(filename, 'r') as hf:
# #     pcls = hf['point_cloud'][:]
#     #[frames, vehicles, points_per_cloud, 4]
#     pcls = hf['point_cloud'][:, vidx ,: , :3]
#     #[frames, points_per_cloud, rgb]
    
# #     pose = hf['lidar_pose'][:]
#     #[frames, vehicles, (x,y,z, rotx, roty, rotz)]
#     pose = hf['lidar_pose'][:, vidx, :]

# c1 = pcls[idx]
# c2 = pcls[idx+1]

# #---------------------------------------------------------------------------------------


# # load custom point cloud geneated in matlab------------------------------------------
# c1 = np.loadtxt("scene2_scan1.txt", dtype = float) #small cylinders
# c2 = np.loadtxt("scene2_scan2.txt", dtype = float)
# # c1 = np.loadtxt("scene3_scan1.txt", dtype = float) #rectangles
# # c2 = np.loadtxt("scene3_scan2.txt", dtype = float)
# # c1 = np.loadtxt("scene4_scan1.txt", dtype = float) #cylinders
# # c2 = np.loadtxt("scene4_scan2.txt", dtype = float)

# # c1 = c1[c1[:,2] > -1.55] #ignore ground plane
# # c2 = c2[c2[:,2] > -1.55] #ignore ground plane

# # x = tf.constant([0., 0.3, 0., 0., 0.0, -0.1])
# # rot = R_tf(x[3:])
# # c2 = c1 @ rot.numpy() + x[:3].numpy()
# # # c2 = (c1 +  x[:3].numpy()) @ rot.numpy()

# #add noise (if not generated when point clouds were created)
# c1 += 0.02*np.random.randn(np.shape(c1)[0], 3)
# c2 += 0.02*np.random.randn(np.shape(c2)[0], 3) 

# # ------------------------------------------------------------------------------------

# # #single distinct cluster---------------------------------------------------------------
# c1 = np.random.randn(3000,3)*tf.constant([0.3,0.04,0.3]) + tf.constant([0.,8.,0.])
# c2 = np.random.randn(3000,3)*tf.constant([0.3,0.04,0.3]) + tf.constant([0.,8.,0.]) - np.array([0.1, 0.1, 0.0])
# # # c2 = c1 - np.array([0.1, 0.3, 0.0])
# # # -------------------------------------------------------------------------------------

#run once to get rough estimate and remove outlier points
# x0 = tf.constant([0.6018, 0.00556, -0.015, 0.0016, 0.0006, -0.01378]) #138
it = ICET(cloud1 = c1, cloud2 = c2, fid = 30, niter = 20, draw = True, group = 2, RM = False)

#run again to re-converge with outliers removed
# cloud1 = it.cloud1_static
# cloud1 = cloud1[cloud1[:,2] > -1.5 ] #remove ground plane 2nd time around
# it = ICET(cloud1 = it.cloud1_static, cloud2 = c2, fid = 70, niter = 20, draw = True, group = 2, RM = False)

# it = ICET(cloud1 = c1, cloud2 = c2, fid = 100, niter = 20, draw = True, group = 2, x0 = it.X)
# it = ICET(cloud1 = c1, cloud2 = c2, fid = 90, niter = 10, draw = True)

# print("\n predicted solution error covariance: \n", it.pred_stds)

ViewInteractiveWidget(it.plt.window)