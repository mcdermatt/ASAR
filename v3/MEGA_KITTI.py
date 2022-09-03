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


num_frames = 4500

basedir = "E:/KITTI/dataset/"
date = "2011_09_26"
drive = '00' #urban
dataset = pykitti.raw(basedir, date, drive)


NDT_estimates = np.zeros([num_frames, 6])
initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])

for i in range(num_frames):

	print("\n ~~~~~~~~~~~~~~~~~~ Frame ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	velo1 = dataset.get_velo(i) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c1 = velo1[:,:3]
	velo2 = dataset.get_velo(i+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c2 = velo2[:,:3]

	# c1 = c1[c1[:,2] > -1.5] #ignore ground plane
	# c2 = c2[c2[:,2] > -1.5] #ignore ground plane
	# c1 = c1[c1[:,2] > -2.] #ignore reflections
	# c2 = c2[c2[:,2] > -2.] #ignore reflections


	it = ICET(cloud1 = c1, cloud2 = c2, fid = 90, niter = 15, draw = False, group = 2, 
		RM = True, DNN_filter = False, x0 = initial_guess)
	
	NDT_estimates[i] = it.X #* (dataset.timestamps[i+1] - dataset.timestamps[i]).microseconds/(10e5)/0.1

	initial_guess = it.X

	#sanity check so things don't explode:
	if initial_guess[0] < 0:
		initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])


	#save text file of point clouds so we can run the other benchmarks with MatLab
	# np.savetxt("E:/KITTI/drive_00_text/scan" + str(i) + ".txt", c1)

	#periodically save so we don't lose everything...
	if i % 100 == 0:
		print("saving...")
		np.savetxt("KITTI_estimates_NDT_spherical_TEST.txt", NDT_estimates)
