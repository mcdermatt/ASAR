"""generates data for training perspective shift filter DNN from KITTI sample dataset"""

import pykitti
import numpy as np
import tensorflow as tf

#limit GPU memory ------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)
#-----------------------------------------------------------------

from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf
from metpy.calc import lat_lon_grid_deltas

numShifts = 10 #5 #number of times to resample and translate each voxel each scan
runLen = 50
npts = 100 #50 

# init KITTI dataset
# basedir = 'C:/kitti/' #windows
basedir = '/media/derm/06EF-127D1/KITTI'
date = '2011_09_26'
# drive = '0005'# urban dataset used in 3D-ICET paper 
# drive = '0091'# urban dataset, 341 frames, shopping center, some pedestians (using up to 250 for train, 250+ for test)
# drive = '0095'# urban dataset, 267 frames, tight road, minimal other vehicles 
# drive = '0117' #mixed, trees and sharp turns for 659 frames
drive = '0071' #1058 frames, crowded shopping center
dataset = pykitti.raw(basedir, date, drive)

for idx in range(runLen):
	velo1 = dataset.get_velo(idx) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c1 = velo1[:,:3]
	velo2 = dataset.get_velo(idx+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c2 = velo2[:,:3]
	c1 = c1[c1[:,2] > -1.5] #ignore ground plane
	c2 = c2[c2[:,2] > -1.5] #ignore ground plane
	# c1 = c1[c1[:,2] > -2.] #ignore reflections
	# c2 = c2[c2[:,2] > -2.] #ignore reflections

	poses0 = dataset.oxts[idx] #<- ID of 1st scan
	poses1 = dataset.oxts[idx+1] #<- ID of 2nd scan
	dt = 0.1037 #mean time between lidar samples
	OXTS_ground_truth = tf.constant([poses1.packet.vf*dt, -poses1.packet.vl*dt, poses1.packet.vu*dt, poses1.packet.wf*dt, poses1.packet.wl*dt, poses1.packet.wu*dt])
	shift_scale = 0.0 #standard deviation by which to shift the grid BEFORE SAMPLING corresponding segments of the point cloud
	shift = tf.cast(tf.constant([shift_scale*tf.random.normal([1]).numpy()[0], shift_scale*tf.random.normal([1]).numpy()[0], 0.2*shift_scale*tf.random.normal([1]).numpy()[0], 0, 0, 0]), tf.float32)

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 2, draw = False, group = 2, 
		RM = True, DNN_filter = False, cheat = OXTS_ground_truth+shift)

	#Get ragged tensor containing all points from each scan inside each sufficient voxel
	in1 = it.inside1
	npts1 = it.npts1
	in2 = it.inside2
	npts2 = it.npts2
	corr = it.corr #indices of bins that have enough points from scan1 and scan2

	#get indices of rag with >= 25 elements
	ncells = tf.shape(corr)[0].numpy() #num of voxels with sufficent number of points
	# print(tf.gather(npts2, corr))
	enough1 = tf.gather(in1, corr)
	enough2 = tf.gather(in2, corr)

	for j in range(numShifts):
		#init array to store indices
		idx1 = np.zeros([ncells ,npts])
		idx2 = np.zeros([ncells ,npts])

		#loop through each element of ragged tensor
		for i in range(ncells):
		    idx1[i,:] = tf.random.shuffle(enough1[i])[:npts].numpy() #shuffle order and take first 25 elements
		    idx2[i,:] = tf.random.shuffle(enough2[i])[:npts].numpy() #shuffle order and take first 25 elements

		idx1 = tf.cast(tf.convert_to_tensor(idx1), tf.int32) #indices in scan 1 of points we've selected
		idx2 = tf.cast(tf.convert_to_tensor(idx2), tf.int32) 

		from1 = tf.gather(it.cloud1_tensor, idx1)
		# from2 = tf.gather(it.cloud2_tensor_OG, idx2) #corresponding points in OG pose
		from2 = tf.gather(it.cloud2_tensor, idx2) 	   #transformed by "ground truth" translation

		scan1 = tf.reshape(from1, [-1, 3]).numpy()
		scan2 = tf.reshape(from2, [-1, 3]).numpy()

		#randomly translate each sample from scan 2
		rand = tf.constant([1., 1., 0.1])*tf.random.normal([ncells, 3])
		#tile and apply to scan2
		t = tf.tile(rand, [npts,1])
		t = tf.reshape(tf.transpose(t), [3,npts,-1])
		t = tf.transpose(t, [2,1,0])
		t = tf.reshape(t, [-1, 3])
		scan2 += t.numpy()

		full_soln_vec = rand + shift[:3]
		compact_soln_vec = it.L @ tf.transpose(it.U, [0,2,1]) @ full_soln_vec[:,:,None] #remove extended axis
		compact_soln_vec = tf.matmul(it.U, compact_soln_vec) #project back to XYZ
		compact_soln_vec = compact_soln_vec[:,:,0] #get rid of extra dimension

		soln = full_soln_vec #consider entire solution vector (compact and extended directions)
		# soln = compact_soln_vec #only consider ground truth solutions in directions deemed useful by ICET

		#initialize array to store data on first iteration
		if idx*(j+1) == 0:
			scan1_cum = scan1
			scan2_cum = scan2
			soln_cum = rand + shift[:3]
		else:
			scan1_cum = np.append(scan1_cum, scan1, axis = 0)
			scan2_cum = np.append(scan2_cum, scan2, axis = 0)
			soln_cum = np.append(soln_cum, soln, axis = 0)

		#test
		# print("\n full_soln_vec:", tf.shape(full_soln_vec))
		# print("\n compact_soln_vec:", tf.shape(compact_soln_vec))
		# print("t", tf.shape(t))
		# print("U_i", tf.shape(it.U))


	print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

#smol
# np.savetxt('perspective_shift/training_data/ICET_KITTI_scan1_25_shifted.txt', scan1_cum)
# np.savetxt('perspective_shift/training_data/ICET_KITTI_scan2_25_shifted.txt', scan2_cum)
# np.savetxt('perspective_shift/training_data/ICET_KITTI_ground_truth_25_shifted.txt', soln_cum)

#big
np.save('/media/derm/06EF-127D1/TrainingData/KITTI_0071_scan1_100pts', scan1_cum)
np.save('/media/derm/06EF-127D1/TrainingData/KITTI_0071_scan2_100pts', scan2_cum)
np.save('/media/derm/06EF-127D1/TrainingData/KITTI_0071_ground_truth_100pts', soln_cum)

#0071v2 - removed moving objects at thresh of 0.1