from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp

#limit GPU memory ------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    memlim = 12*1024
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memlim)])
  except RuntimeError as e:
    print(e)
#-----------------------------------------------------------------

from ICET_spherical import ICET
from metpy.calc import lat_lon_grid_deltas


num_frames = 799

basepath = '/media/derm/06EF-127D2/KITTI'
date = "2011_09_26"
# drive = '00' #urban
drive = '03' #trees
#drive = '09' #small town + woods on steep hill
dataset = pykitti.odometry(basepath, drive)

ICET_estimates = np.zeros([num_frames, 6])
initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])
before_correction = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

for i in range(num_frames):

	print("\n ~~~~~~~~~~~~~~~~~~ Frame ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	velo1 = dataset.get_velo(i) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c1 = velo1[:,:3]
	velo2 = dataset.get_velo(i+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c2 = velo2[:,:3]

	c1 = c1[c1[:,2] > -1.5] #ignore ground plane
	c2 = c2[c2[:,2] > -1.5] #ignore ground plane
	# c1 = c1[c1[:,2] > -2.] #ignore reflections
	# c2 = c2[c2[:,2] > -2.] #ignore reflections

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 5, draw = False, group = 2, 
		RM = False, DNN_filter = False, x0 = initial_guess)
	
	ICET_estimates[i] = it.X
	before_correction[i] = it.before_correction
	ICET_pred_stds[i] = it.pred_stds

	# initial_guess = it.X
	#sanity check so things don't explode:
	# if initial_guess[0] < 0:
	# 	initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])


	#save text file of point clouds so we can run the other benchmarks with MatLab
	# np.savetxt("E:/KITTI/drive_00_text/scan" + str(i) + ".txt", c1)

	# #periodically save so we don't lose everything...
	# if i % 10 == 0:
	# 	print("saving...")
	# 	# np.savetxt("perspective_shift/sim_results/KITTI_03_raw.txt", before_correction)
	# 	np.savetxt("perspective_shift/sim_results/KITTI_03_2sigma.txt", ICET_estimates)
	# 	# np.savetxt("perspective_shift/sim_results/KITTI_03_noDNN.txt", before_correction)
	# 	# np.savetxt("perspective_shift/sim_results/KITTI_03_CompactNet.txt", ICET_estimates)
	# 	# np.savetxt("perspective_shift/sim_results/KITTI_03_pred_stds.txt", ICET_pred_stds)
