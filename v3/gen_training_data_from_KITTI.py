import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf
from metpy.calc import lat_lon_grid_deltas


numShifts = 5 #number of times to resample and translate each voxel each scan
runLen = 15 #150
npts = 50 #25

# init KITTI dataset
basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'# urban dataset used in 3D-ICET paper 
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


	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 12, draw = False, group = 2, 
		RM = True, DNN_filter = False, cheat = OXTS_ground_truth)

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

		if idx*(j+1) == 0:
			scan1_cum = scan1
			scan2_cum = scan2
			rand_cum = rand
		else:
			scan1_cum = np.append(scan1_cum, scan1, axis = 0)
			scan2_cum = np.append(scan2_cum, scan2, axis = 0)
			rand_cum = np.append(rand_cum, rand, axis = 0)

	print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

#smol
np.savetxt('perspective_shift/training_data/ICET_KITTI_scan1_50.txt', scan1_cum)
np.savetxt('perspective_shift/training_data/ICET_KITTI_scan2_50.txt', scan2_cum)
np.savetxt('perspective_shift/training_data/ICET_KITTI_ground_truth_50.txt', rand_cum)

#big
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_scan1_50.txt', scan1_cum)
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_scan2_50.txt', scan2_cum)
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_ground_truth_50.txt', rand_cum)
