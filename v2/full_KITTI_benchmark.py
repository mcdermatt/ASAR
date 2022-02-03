import numpy as np
import tensorflow as tf
import matplotlib.pyplot as PLT

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
from utils import R2Euler


nc = 5	 #number of cycles
mnp = 50 #50 #minimum number of points per voxel
D = False #draw sim
DG = False #draw grid
DE = False #draw ellipsoids
DC = False #draw correspondences
TD = False #use test dataset
CM = "voxel" #correspondence method, "voxel" or "NN"
vizL = False #draw arrows in direction of non-truncated directions for each distribution

plt = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)

## Use benchmark data ----------------------------------------------------------
basepath = "E:/KITTI/dataset/"
# sequence = '05'
sequence = '00'

dataset = pykitti.odometry(basepath, sequence)
#-------------------------------------------------------------------------------

#temp plot for debug--------
fig = PLT.figure()
ax = fig.add_subplot()
# ax.set_aspect('equal')
#---------------------------

# num_frames = 2760 #05
# num_frames = 4540 #00
num_frames = 50

E_hist_true = np.zeros([num_frames, 1])
N_hist_true = np.zeros([num_frames, 1])
true_pose = np.zeros([num_frames, 6])

ICET_estimates = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

start = 0

for i in range(start, num_frames):
	print("~~~~~~~~~~~~~~~~~ Frame #", i, "~~~~~~~~~~~~~~~~~~~~~~~~")

	#___Use ICET to get pos and accuracy estimates______________________
	velo1 = dataset.get_velo(i) # Each scan is a Nx4 array of [x,y,z,reflectance]
	cloud1 = velo1[:,:3]
	limy = 30 #horizotal lim in +/- directions
	lim2 = -1 #-1.5 #IGNORE POINTS ON GROUND!!
	cloud1 = cloud1[ cloud1[:,1] < limy]
	cloud1 = cloud1[ cloud1[:,1] > -limy]
	cloud1 = cloud1[ cloud1[:,2] > lim2]
	cloud1_tensor = tf.convert_to_tensor(cloud1, np.float32)

	velo2 = dataset.get_velo(i+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	cloud2 = velo2[:,:3]
	cloud2 = cloud2[ cloud2[:,1] < limy]
	cloud2 = cloud2[ cloud2[:,1] > -limy]
	cloud2 = cloud2[ cloud2[:,2] > lim2]
	cloud2_tensor = tf.convert_to_tensor(cloud2, np.float32)

	# f = tf.constant([20,20,2]) #need larger voxel sizes for 018
	# lim = tf.constant([-100.,100.,-100.,100.,-10.,10.]) #needs to encompass every point

	#forward direction more important than sideways
	f = tf.constant([40,20,2]) 
	lim = tf.constant([-100.,100.,-50.,50.,-5.,5.])

	#estimate solution vector x using ICET
	if i == start:
		Q, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = lim, 
			fid = f, num_cycles = nc , min_num_pts = mnp, draw = D)
	# use estimates from previous frames to initialize xHat0
	else:
		Q, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = lim, 
			fid = f, num_cycles = nc , min_num_pts = mnp, draw = D, xHat0 = x_hist[-1])
		# #run a 2nd time using initial conditions from output of coarse voxelization
		# f2 = tf.constant([50,50,2])
		# Q2, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = lim, 
		# 	fid = f2, num_cycles = 3 , min_num_pts = mnp, draw = D, xHat0 = x_hist[-1])

	ICET_estimates[i] = x_hist[-1].numpy()
	ICET_pred_stds[i,:] = np.sqrt(abs(np.array([Q[0,0], Q[1,1], Q[2,2], Q[3,3], Q[4,4], Q[5,5]]))) 
	#_________________________________________________________________________

	E_hist_true[i] = dataset.poses[i][0,3]
	N_hist_true[i] = dataset.poses[i][2,3]
	
	# body_frame_delta = dataset.poses[i][:3,:3].T.dot( dataset.poses[i+1][:3,3] - dataset.poses[i][:3,3])
	body_frame_delta = dataset.poses[i+1][:3,:3].T.dot( dataset.poses[i+1][:3,3] - dataset.poses[i][:3,3])

	# print(body_frame_delta)
	true_pose[i,0] = body_frame_delta[2]
	true_pose[i,1] = body_frame_delta[0]
	true_pose[i,2] = -body_frame_delta[1]

	angs0 = R2Euler(dataset.poses[i][:3,:3])
	angs1 = R2Euler(dataset.poses[i+1][:3,:3])
	true_pose[i,4] = angs1[0] - angs0[0]
	true_pose[i,5] = angs1[1] - angs0[1]
	true_pose[i,3] = -angs1[2] + angs0[2]

	# print("estimated: \n", x_hist[-1].numpy)
	print("true pose: \n", true_pose[i])

np.savetxt("ICET_benchmark_00.txt", ICET_estimates)
np.savetxt("ICET_pred_stds_00.txt", ICET_pred_stds)
np.savetxt("true_pose_00.txt", true_pose)

ax.plot(E_hist_true,N_hist_true)
# ax.plot(true_pose[:,0])
# ax.plot(ICET_estimates[:,0])

PLT.show()