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
from metpy.calc import lat_lon_grid_deltas


""" Runs ICET on each sequential set of scans in the KITTI dataset """


nc = 5	 #number of iterations of ICET per each pair of clouds
mnp = 50 #minimum number of points per voxel
D = False #draw sim

# plt = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)
plt = Plotter(N=1, axes=4, interactive=True)

## set up point clouds
basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005' #city
# drive = '0009' #suburban
# drive = '0018' #difficult intersection case
dataset = pykitti.raw(basedir, date, drive)
# f = tf.constant([50,50,2]) #fidelity in x, y, z # < 5s  --- works for 0005
# f = tf.constant([20,20,2]) #0018
f = tf.constant([40,40,4]) #test

lim = tf.constant([-100.,100.,-100.,100.,-10.,10.]) #needs to encompass every point
npts = 100000 #need to cut number of points at finer voxel sizes because I only have 3gb VRAM
# npts = 123397 

# num_frames = 20 #debug
num_frames = 150 #0005
# num_frames = 268 # 0018
# num_frames = 445 #0009
ICET_estimates = np.zeros([num_frames, 6])
OXTS_baseline = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

for i in range(num_frames):
	print("~~~~~~~~~~~~~~~~~ Frame #", i, "~~~~~~~~~~~~~~~~~~~~~~~~")

	velo1 = dataset.get_velo(i) # Each scan is a Nx4 array of [x,y,z,reflectance]
	cloud1 = velo1[:,:3]
	cloud1_tensor = tf.convert_to_tensor(cloud1, np.float32)
	velo2 = dataset.get_velo(i+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	cloud2 = velo2[:,:3]
	cloud2_tensor = tf.convert_to_tensor(cloud2, np.float32)

	#estimate solution vector x using ICET
	if i == 0:
		Q, x_hist = ICET3D(cloud1_tensor[:npts], cloud2_tensor[:npts], plt, bounds = lim, 
			fid = f, num_cycles = nc , min_num_pts = mnp, draw = D)
	# use estimates from previous frames to initialize xHat0
	else:
		Q, x_hist = ICET3D(cloud1_tensor[:npts], cloud2_tensor[:npts], plt, bounds = lim, 
			fid = f, num_cycles = nc , min_num_pts = mnp, draw = D, xHat0 = x_hist[-1])

	ICET_estimates[i] = x_hist[-1].numpy()

	ICET_pred_stds[i,:] = np.sqrt(abs(np.array([Q[0,0], Q[1,1], Q[2,2], Q[3,3], Q[4,4], Q[5,5]]))) 

	#get transformations in frame of OXTS GPS/INS sensor
	poses0 = dataset.oxts[i] #<- ID of 1st scan
	poses1 = dataset.oxts[i+1] #<- ID of 2nd scan
	lat0 = poses0.packet.lat
	lon0 = poses0.packet.lon
	alt0 = poses0.packet.alt
	lat1 = poses1.packet.lat
	lon1 = poses1.packet.lon
	alt1 = poses1.packet.alt

	#these are "pint" objects which hold on to units
	dx_oxts, dy_oxts = lat_lon_grid_deltas(np.array([lon0,lon1]), np.array([lat0, lat1]))
	# print(dx_oxts, dy_oxts) 
	dx_oxts = dx_oxts[0,0].magnitude
	dy_oxts = dy_oxts[0,0].magnitude
	dz_oxts = (alt0-alt1)
	droll_oxts = (poses0.packet.roll - poses1.packet.roll)
	dpitch_oxts = (poses0.packet.pitch - poses1.packet.pitch)
	dyaw_oxts = (poses0.packet.yaw - poses1.packet.yaw)

	#get rotation matrix to bring things into frame of lidar unit
	# rot = poses0.T_w_imu[:3,:3] #ignore reflectance componenet
	rot = poses1.T_w_imu[:3,:3] #trying this

	dxyz_oxts = np.array([[dx_oxts, dy_oxts, dz_oxts]])
	dxyz_lidar = dxyz_oxts.dot(rot)

	# using lat/ lon deltas
	# OXTS_baseline[i] = np.array([[dxyz_lidar[0,0], dxyz_lidar[0,1], dxyz_lidar[0,2], droll_oxts, dpitch_oxts, dyaw_oxts]])

	#using velocity
	# dt = (dataset.timestamps[i+1] - dataset.timestamps[i]).microseconds/(10e5)
	dt = 0.10327 #mean time between lidar samples
	# OXTS_baseline[i] = np.array([[poses1.packet.vf*dt, poses1.packet.vl*dt, poses1.packet.vu*dt, droll_oxts, dpitch_oxts, dyaw_oxts]]) #works, but has stepping behavior for yaw
	OXTS_baseline[i] = np.array([[poses1.packet.vf*dt, poses1.packet.vl*dt, poses1.packet.vu*dt, -poses1.packet.wf*dt, -poses1.packet.wl*dt, -poses1.packet.wu*dt]]) #test


	print("\n solution from ICET \n", ICET_estimates[i])
	print("\n solution from GPS/INS \n", OXTS_baseline[i])


# ans = np.sqrt(abs(Q.numpy())) 
# print("\n predicted solution standard deviation of error: \n", ans[0,0], ans[1,1], ans[2,2], ans[3,3], ans[4,4], ans[5,5])

print("ICET_estimates \n", ICET_estimates)
print("\n OXTS baseline \n", OXTS_baseline)

np.savetxt("ICET_pred_stds_926_0005.txt", ICET_pred_stds)
np.savetxt("ICET_estimates_926_0005.txt", ICET_estimates)
np.savetxt("OXTS_baseline_926_0005.txt", OXTS_baseline)

#NOTES:
#		test3 == [20,20,2], xHat0 initialized at zero, n=5, mnp = 50
#		test4 == [50,50,5], ||	, n=5, mnp = 50
#		...0005.txt == [40,40,4], 100k pts, n = 5, mnp = 50