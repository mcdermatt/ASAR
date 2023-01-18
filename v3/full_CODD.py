from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf

#limit GPU memory ------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    memlim = 4*1024
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memlim)])
  except RuntimeError as e:
    print(e)
#-----------------------------------------------------------------

from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from metpy.calc import lat_lon_grid_deltas
import h5py


num_frames = 124 #124
vidx = 0 #vehicle index

# filename = 'C:/CODD/data/m2v7p3s333.hdf5'
filename = '/media/derm/06EF-127D2/CODD/data/m10v11p6s30.hdf5' #wide road, palm trees, and traffic (used in 3D paper)

with h5py.File(filename, 'r') as hf:
#     pcls = hf['point_cloud'][:]
	#[frames, vehicles, points_per_cloud, 4]
	pcls = hf['point_cloud'][:, vidx ,: , :3]
	#[frames, points_per_cloud, rgb]

#     pose = hf['lidar_pose'][:]
	#[frames, vehicles, (x,y,z, rotx, roty, rotz)]
	pose = hf['lidar_pose'][:, vidx, :]


ICET_estimates = np.zeros([num_frames, 6])
CODD_baseline = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

intial_guess = tf.constant([0., 0., 0., 0., 0., 0.])

for i in range(num_frames):

	print("\n ~~~~~~~~~~~~~~~~~~ Epoch ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")
	c1 = pcls[i]
	c2 = pcls[i+1]

	noise_scale = 0.005 #0.01
	c1 += noise_scale*np.random.randn(np.shape(c1)[0], 3)
	c2 += noise_scale*np.random.randn(np.shape(c2)[0], 3)

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 122, niter = 7, draw = False, group = 2, RM = True, DNN_filter = False, x0 = intial_guess)

	ICET_pred_stds[i] = it.pred_stds
	ICET_estimates[i] = it.X #* (dataset.timestamps[i+1] - dataset.timestamps[i]).microseconds/(10e5)/0.1
	intial_guess = it.X

	print("\n solution from ICET \n", ICET_estimates[i])
	print("\n pred_stds \n", it.pred_stds)
	# print("\n ground truth transformation \n", np.diff(pose, axis = 0)[i])


np.savetxt("results/ICET_pred_stds_CODD_fid122.txt", ICET_pred_stds)
np.savetxt("results/ICET_estimates_CODD_fid122.txt", ICET_estimates)

# np.savetxt("results/NDT_pred_stds_CODD_v3.txt", ICET_pred_stds)
# np.savetxt("results/NDT_estimates_CODD_v3.txt", ICET_estimates)

#v3 - basic outlier exclusion
#v4 - using dnn filter
#v5 - 7/14 after corrections to U matrix, no dnn
#v6 - 7/14, again
#v7 - 7/31, with DNN

