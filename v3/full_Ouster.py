from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import pickle
import pandas as pd

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

from ICET_spherical import ICET
from metpy.calc import lat_lon_grid_deltas
import trimesh

num_frames = 715
ICET_estimates = np.zeros([num_frames, 6])
OXTS_baseline = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])
before_correction = np.zeros([num_frames, 6])
initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])

for i in range(num_frames):
# for i in range(0,num_frames,3): #skip3

	print("\n ~~~~~~~~~~~~~~~~~~ Epoch ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	prefix = "/media/derm/06EF-127D2/Ouster/csv/pcap_out_"
	fn1 = prefix + '%06d.csv' %(i)
	df1 = pd.read_csv(fn1, sep=',', skiprows=[0])
	pc1 = df1.values[:,8:11]*0.001

	fn2 = prefix + '%06d.csv' %(i+1)
	df2 = pd.read_csv(fn2, sep=',', skiprows=[0])
	pc2 = df2.values[:,8:11]*0.001

	# pc1 = pc1[pc1[:,2] > -1.25] #ignore ground plane
	# pc2 = pc2[pc2[:,2] > -1.25] #ignore ground plane

	it = ICET(cloud1 = pc1, cloud2 = pc2, fid = 80, niter = 10, 
		draw = False, group = 2, RM = True, DNN_filter = False, x0 = initial_guess)

	ICET_estimates[i] = it.X #* (dataset.timestamps[i+1] - dataset.timestamps[i]).microseconds/(10e5)/0.1
	before_correction[i] = it.before_correction
	ICET_pred_stds[i] = it.pred_stds

	initial_guess = it.X

	# # -------------------------------
	# dt = 0.1
	# OXTS_baseline[i] = np.array([[poses1.packet.vf*dt, poses1.packet.vl*dt, poses1.packet.vu*dt, -poses1.packet.wf*dt, -poses1.packet.wl*dt, -poses1.packet.wu*dt]]) #test
	# print("\n solution from GPS/INS \n", OXTS_baseline[i])
	# # -------------------------------

	#periodically save so we don't lose everything...
	if i % 10 == 0:
		print("saving...")
		np.savetxt("ouster_ICET_estimates.txt", ICET_estimates)
		np.savetxt("ouster_ICET_pred_stds.txt", ICET_pred_stds)

# np.savetxt("perspective_shift/sim_results/KITTI_0028_noDNN.txt", before_correction)
np.savetxt("ouster_ICET_estimates.txt", ICET_estimates)
np.savetxt("ouster_ICET_pred_stds.txt", ICET_pred_stds)
# np.savetxt("perspective_shift/sim_results/KITTI_0028_OXTS_baseline_gps.txt", OXTS_baseline)