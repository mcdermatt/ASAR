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


nc = 10	 #number of iterations of ICET per each pair of clouds
mnp = 50 #minimum number of points per voxel
D = False #draw sim

# plt = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)
plt = Plotter(N=1, axes=4, interactive=True)

## set up point clouds
basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'
dataset = pykitti.raw(basedir, date, drive)
f = tf.constant([50,50,2]) #fidelity in x, y, z # < 5s
lim = tf.constant([-100.,100.,-100.,100.,-10.,10.]) #needs to encompass every point
npts = 100000

num_frames = 25
ICET_estimates = np.zeros([num_frames, 6])
OXTS_baseline = np.zeros([num_frames, 6])

for i in range(num_frames):

	velo1 = dataset.get_velo(i) # Each scan is a Nx4 array of [x,y,z,reflectance]
	cloud1 = velo1[:,:3]
	cloud1_tensor = tf.convert_to_tensor(cloud1, np.float32)
	velo2 = dataset.get_velo(i+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	cloud2 = velo2[:,:3]
	cloud2_tensor = tf.convert_to_tensor(cloud2, np.float32)

	#estimate solution vector x using ICET
	Q, x_hist = ICET3D(cloud1_tensor[:npts], cloud2_tensor[:npts], plt, bounds = lim, 
		fid = f, num_cycles = nc , min_num_pts = mnp, draw = D)
	ICET_estimates[i] = x_hist[-1].numpy()

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
	rot = poses0.T_w_imu[:3,:3] #ignore reflectance componenet

	dxyz_oxts = np.array([[dx_oxts, dy_oxts, dz_oxts]])
	dxyz_lidar = dxyz_oxts.dot(rot)

	print("\n solution from GPS/INS \n",dxyz_lidar, droll_oxts, dpitch_oxts, dyaw_oxts)

	OXTS_baseline[i] = np.array([[dxyz_lidar[0,0], dxyz_lidar[0,1], dxyz_lidar[0,2], droll_oxts, dpitch_oxts, dyaw_oxts]])

# ans = np.sqrt(abs(Q.numpy())) 
# print("\n predicted solution standard deviation of error: \n", ans[0,0], ans[1,1], ans[2,2], ans[3,3], ans[4,4], ans[5,5])

print("ICET_estimates \n", ICET_estimates)
print("\n OXTS baseline \n", OXTS_baseline)

np.savetxt("ICET_estimates.txt", ICET_estimates)
np.savetxt("OXTS_baseline.txt", OXTS_baseline)