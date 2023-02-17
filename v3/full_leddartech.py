from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import pickle

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
from pioneer.das.api.platform import Platform
from scipy.spatial.transform import Rotation as R

# drive = "20200721_144638_part36_1956_2229" #old church (used in 3D paper)
# num_frames = 274
# drive = "20200706_161206_part22_670_950" #subrubs
# num_frames = 274
drive = "20200618_191030_part17_1120_1509" #long straight stretch, passing cyclists
num_frames = 388

#new
dataset_path = "/media/derm/06EF-127D3/leddartech/" + drive
config_path = "/media/derm/06EF-127D3/leddartech/" + drive + "/platform.yml"
pf = Platform(dataset_path, config_path)

ICET_estimates = np.zeros([num_frames, 6])
OXTS_baseline = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])
before_correction = np.zeros([num_frames, 6])
# initial_guess = tf.constant([1.0, 0., 0., 0., 0., 0.])


for idx in range(num_frames):
# for i in range(0,num_frames,3): #skip3

	print("\n ~~~~~~~~~~~~~~~~~~ Epoch ",  idx," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	#new -----------------------------------------------------------------------

	data1 = pf['ouster64_bfc_xyzit'][idx].get_point_cloud(undistort = True)
	data2 = pf['ouster64_bfc_xyzit'][idx+1].get_point_cloud(undistort = True)
	ts_lidar = pf['ouster64_bfc_xyzit'][idx].timestamp

	#old -----------------------------------------------------------------------
	# prefix = "/media/derm/06EF-127D2/leddartech/" + drive + "/ouster64_bfc_xyzit/" 
	# fn1 = prefix + '%08d.pkl' %(i)
	# with open(fn1, 'rb') as f:
	# 	data1 = pickle.load(f)
	# data1 = np.asarray(data1.tolist())[:,:3]

	# fn2 = prefix + '%08d.pkl' %(i+1)
	# with open(fn2, 'rb') as f:
	# 	data2 = pickle.load(f)
	# data2 = np.asarray(data2.tolist())[:,:3]

	# --------------------------------------------------------------------------


	#get ground truth from GNSS data-------------------------------------------
	GNSS = pf.sensors['sbgekinox_bcc']
	from pioneer.das.api.egomotion.imu_egomotion_provider import IMUEgomotionProvider as emp 
	name = pf
	test = emp(name, GNSS['navposvel'], GNSS['ekfeuler'])
	timestamps = test.get_timestamps()

	gt_vec = np.zeros([len(timestamps)-1,6])
	for i in range(1,len(timestamps)):
		#get translations from GNSS/INS baseline
		gt_vec[i-1,0] = test.get_transform(timestamps[i])[1,3] - test.get_transform(timestamps[i-1])[1,3]
		gt_vec[i-1,1] = test.get_transform(timestamps[i])[0,3] - test.get_transform(timestamps[i-1])[0,3]
		gt_vec[i-1,2] = test.get_transform(timestamps[i])[2,3] - test.get_transform(timestamps[i-1])[2,3]
		#get rotations
		T1 = test.get_transform(timestamps[i-1])
		T2 = test.get_transform(timestamps[i])
		r1 = R.from_matrix(T1[:3,:3])
		r2 = R.from_matrix(T2[:3,:3])
		gt_vec[i-1,3:] = (r2.as_euler('xyz', degrees=False) - r1.as_euler('xyz', degrees=False))
	vf = np.sqrt(gt_vec[:,0]**2 + gt_vec[:,1]**2)
	gt_vec[:,0] = vf
	gt_vec[:,1] = 0
	gt_vec[:,2] = 0
	gt_vec = gt_vec * 5

	# loop through all GNSS timestamps, stop when larger than ts_lidar and use previous index
	for c in range(len(timestamps)):
		ts_gnss = timestamps[c]
		if ts_gnss > ts_lidar:
			break
	initial_guess = tf.convert_to_tensor(gt_vec[c], dtype = tf.float32)
	# --------------------------------------------------------------------------


	# data1 = data1[data1[:,2] > -0.75] #ignore ground plane
	# data2 = data2[data2[:,2] > -0.75] #ignore ground plane

	# it = ICET(cloud1 = data1, cloud2 = data2, fid = 70, niter = 2, 
	# 	draw = True, group = 2, RM = True, DNN_filter = False, x0 = initial_guess)
	it = ICET(cloud1 = data1, cloud2 = data2, fid = 70, niter = 2, 
		draw = True, group = 2, RM = True, DNN_filter = False, cheat = initial_guess)


	ICET_estimates[idx] = it.X #* (dataset.timestamps[i+1] - dataset.timestamps[i]).microseconds/(10e5)/0.1
	before_correction[idx] = it.before_correction
	ICET_pred_stds[idx] = it.pred_stds

	# initial_guess = it.X #use last estimate

	screenshot('demo/shaded_residuals/test_%03d.png' %idx)
	it.plt.close()

	#periodically save so we don't lose everything...
	if idx % 10 == 0:
		print("saving...")
		np.savetxt("results/leddartech_ICET_estimates_suburb_undistorted.txt", ICET_estimates)
		np.savetxt("results/leddartech_ICET_pred_stds_suburb_undistorted.txt", ICET_pred_stds)

# np.savetxt("perspective_shift/sim_results/KITTI_0028_noDNN.txt", before_correction)
np.savetxt("results/leddartech_ICET_estimates_suburb_undistorted.txt", ICET_estimates)
np.savetxt("results/leddartech_ICET_pred_stds_suburb_undistorted.txt", ICET_pred_stds)
# np.savetxt("perspective_shift/sim_results/KITTI_0028_OXTS_baseline_gps.txt", OXTS_baseline)