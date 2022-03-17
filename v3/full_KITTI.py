from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET


num_frames = 20

basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'
frame_range = range(150, 151, 1)
dataset = pykitti.raw(basedir, date, drive)

ICET_estimates = np.zeros([num_frames, 6])
OXTS_baseline = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

intial_guess = tf.constant([0., 0., 0., 0., 0., 0.])

for i in range(num_frames):

	print("\n ~~~~~~~~~~~~~~~~~~ Epoch ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	velo1 = dataset.get_velo(i) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c1 = velo1[:,:3]
	velo2 = dataset.get_velo(i+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c2 = velo2[:,:3]
	# c1 = c1[c1[:,2] > -1.5] #ignore ground plane
	# c2 = c2[c2[:,2] > -1.5] #ignore ground plane
	c1 = c1[c1[:,2] > -2.] #ignore reflections
	c2 = c2[c2[:,2] > -2.] #ignore reflections

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 8, draw = False, group = 2) #, x0 = intial_guess)
	it = ICET(cloud1 = c1, cloud2 = c2, fid = 100, niter = 8, draw = False, group = 2, x0 = it.X)
	it = ICET(cloud1 = c1, cloud2 = c2, fid = 150, niter = 5, draw = False, group = 2, x0 = it.X)

	ICET_estimates[i] = it.X
	ICET_pred_stds[i] = it.pred_stds

	intial_guess = it.X

	poses0 = dataset.oxts[i] 
	poses1 = dataset.oxts[i+1]
	dt = 0.1
	OXTS_baseline[i] = np.array([[poses1.packet.vf*dt, poses1.packet.vl*dt, poses1.packet.vu*dt, -poses1.packet.wf*dt, -poses1.packet.wl*dt, -poses1.packet.wu*dt]]) #test

	print("\n solution from ICET \n", ICET_estimates[i])
	print("\n solution from GPS/INS \n", OXTS_baseline[i])

np.savetxt("ICET_pred_stds_v4.txt", ICET_pred_stds)
np.savetxt("ICET_estimates_v4.txt", ICET_estimates)
np.savetxt("OXTS_baseline_v4.txt", OXTS_baseline)

#v3 - using new clustering 30-100-150, 
#v4 - swaped order of rotate-translate