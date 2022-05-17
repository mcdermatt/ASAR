from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from metpy.calc import lat_lon_grid_deltas


num_frames = 20

basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'
frame_range = range(150, 151, 1)
dataset = pykitti.raw(basedir, date, drive)

ICET_estimates = np.zeros([num_frames, 6])
OXTS_baseline = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])

for i in range(num_frames):

	print("\n ~~~~~~~~~~~~~~~~~~ Epoch ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	# # normal KITTI data~~~~~~
	velo1 = dataset.get_velo(i) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c1 = velo1[:,:3]
	velo2 = dataset.get_velo(i+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c2 = velo2[:,:3]
	# #~~~~~~~~~~~~~~~~~~~~~~~~

	# # Raw KITTI data ~~~~~~~~~~
	# fn1 = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/%010d.txt" %(i)
	# fn2 = "C:/kitti/2011_09_26/2011_09_26_drive_0005_raw/velodyne_points/data/%010d.txt" %(i+1)
	# c1 = np.loadtxt(fn1)[:,:3]
	# c2 = np.loadtxt(fn2)[:,:3]
	# #~~~~~~~~~~~~~~~~~~~~~~~~


	# c1 = c1[c1[:,2] > -1.5] #ignore ground plane
	# c2 = c2[c2[:,2] > -1.5] #ignore ground plane
	# c1 = c1[c1[:,2] > -2.] #ignore reflections
	# c2 = c2[c2[:,2] > -2.] #ignore reflections

	# it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 20, draw = False, group = 2) #, x0 = intial_guess)

	#-------------------------------------------------------------------------------------------------
	#run once to get rough estimate and remove outlier points
	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 17, draw = False, group = 2, 
		RM = True, DNN_filter = True, x0 = initial_guess)
	ICET_pred_stds[i] = it.pred_stds

	#run again to re-converge with outliers removed
	# it = ICET(cloud1 = it.cloud1_static, cloud2 = c2, fid = 70, niter = 20, draw = False, group = 2, RM = False)
	#-------------------------------------------------------------------------------------------------

	ICET_estimates[i] = it.X #* (dataset.timestamps[i+1] - dataset.timestamps[i]).microseconds/(10e5)/0.1
	# ICET_pred_stds[i] = it.pred_stds

	initial_guess = it.X

	# -------------------------------
	poses0 = dataset.oxts[i] 
	poses1 = dataset.oxts[i+1]
	dt = 0.1
	# dt = (dataset.timestamps[i+1] - dataset.timestamps[i]).microseconds/(10e5)
	OXTS_baseline[i] = np.array([[poses1.packet.vf*dt, poses1.packet.vl*dt, poses1.packet.vu*dt, -poses1.packet.wf*dt, -poses1.packet.wl*dt, -poses1.packet.wu*dt]]) #test
	# -------------------------------


	# #-------------------------------
	# #get transformations in frame of OXTS GPS/INS sensor
	# poses0 = dataset.oxts[i] #<- ID of 1st scan
	# poses1 = dataset.oxts[i+1] #<- ID of 2nd scan
	# lat0 = poses0.packet.lat
	# lon0 = poses0.packet.lon
	# alt0 = poses0.packet.alt
	# lat1 = poses1.packet.lat
	# lon1 = poses1.packet.lon
	# alt1 = poses1.packet.alt

	# #these are "pint" objects which hold on to units
	# dx_oxts, dy_oxts = lat_lon_grid_deltas(np.array([lon0,lon1]), np.array([lat0, lat1]))
	# # print(dx_oxts, dy_oxts) 
	# dx_oxts = dx_oxts[0,0].magnitude
	# dy_oxts = dy_oxts[0,0].magnitude
	# dz_oxts = (alt0-alt1)
	# droll_oxts = (poses0.packet.roll - poses1.packet.roll)
	# dpitch_oxts = (poses0.packet.pitch - poses1.packet.pitch)
	# dyaw_oxts = (poses0.packet.yaw - poses1.packet.yaw)

	# #get rotation matrix to bring things into frame of lidar unit
	# # rot = poses0.T_w_imu[:3,:3] #ignore reflectance componenet
	# rot = poses1.T_w_imu[:3,:3] #trying this

	# dxyz_oxts = np.array([[dx_oxts, dy_oxts, dz_oxts]])
	# dxyz_lidar = dxyz_oxts.dot(rot)

	# dt = (dataset.timestamps[i+1] - dataset.timestamps[i]).microseconds/(10e5)

	# # using lat/ lon deltas
	# OXTS_baseline[i] = np.array([[dxyz_lidar[0,0], dxyz_lidar[0,1], dxyz_lidar[0,2], droll_oxts, dpitch_oxts, dyaw_oxts]])/dt*0.1
	# #-------------------------------


	# print("\n solution from ICET \n", ICET_estimates[i])
	print("\n solution from GPS/INS \n", OXTS_baseline[i])

np.savetxt("ICET_pred_stds_v16.txt", ICET_pred_stds)
np.savetxt("ICET_estimates_v16.txt", ICET_estimates)
np.savetxt("OXTS_baseline_v16.txt", OXTS_baseline)

# np.savetxt("OXTS_baseline_gps.txt", OXTS_baseline)

#v3 - using new clustering 30-100-150, 
#v4 - with moving objects removed {50}, with ground plane, sigma thresh = 2
#v5 - same as above, no ground plane
#v6 - running ICET twice, 2nd time around ignoring all points in scan 2 inside moving voxels, NO GROUND PLANE
#v7 - running twice again, with gp, fid = 100, sigma x,y = 2
#v8 - x, y and vertical rotation exclusion- improved results with yaw (almost where we need to be!), not good enough yet
#v9 - same as above, fid = 100s
#v10 - short, first try with DNN
#v11 - DNN full length, some optimization
#v12 - no ground plane, more optimization
#v13 - used in presentation, more compact but obvious low frequency error
#v14 - debugging it/dnn compact, test flipping sign of DNN correction
#v15 - using raw data (rolling shutter effect uncompensated)
#v16 - normal data, using DNN solutions in place of dz