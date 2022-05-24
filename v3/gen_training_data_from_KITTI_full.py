"""generates data for training perspective shift filter DNN from KITTI sample dataset"""

import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf, R2Euler
from metpy.calc import lat_lon_grid_deltas

numShifts = 5 #number of times to resample and translate each voxel each scan
runLen = 400 #4500
npts = 25 #50 

# init KITTI dataset
basedir = "E:/KITTI/dataset/"
date = '2011_09_26'
drive = '00'# long urban training dataset
dataset = pykitti.raw(basedir, date, drive)

#load transformation matrices
filename = "E:/KITTI/dataset/2011_09_26/2011_09_26_drive_00_sync/poses/00.txt"
full_poses = np.loadtxt(filename)
mat_full = np.reshape(full_poses, [-1,3,4])

Rmat = tf.convert_to_tensor(mat_full[:,:,:3])
euls = R2Euler(Rmat)

for idx in range(runLen):

	print("\n ~~~~~~~~~~~~~~~~~~ Scan ",  idx," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	velo1 = dataset.get_velo(idx) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c1 = velo1[:,:3]
	velo2 = dataset.get_velo(idx+1) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c2 = velo2[:,:3]

	# c1 = c1[c1[:,2] > -1.5] #ignore ground plane
	# c2 = c2[c2[:,2] > -1.5] #ignore ground plane

	#get change in rotation
	drot = euls[:,idx+1] - euls[:,idx]
	drot = np.array([-drot[2], drot[0], drot[1] ]) #re-order to match ICET output
	#get translation in vehicle body frame 
	dpos_xyz = mat_full[idx+1,:,3] - mat_full[idx,:,3]
	dpos_bf = np.array([np.sqrt(dpos_xyz[0]**2 + dpos_xyz[2]**2), 0, dpos_xyz[1]])

	OXTS_ground_truth = np.append(dpos_bf, drot)
	#assume zero vertical movement between frames???
	OXTS_ground_truth[2] = 0
	OXTS_ground_truth = tf.cast(tf.convert_to_tensor(OXTS_ground_truth), tf.float32)

	#randomly switch order of scans to minimize bias ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	roll = np.random.rand()
	if roll > 0.5: 
		c1_temp = c1
		c1 = c2
		c2 = c1_temp
		OXTS_ground_truth = -OXTS_ground_truth
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	shift_scale = 0.003 #standard deviation by which to shift the grid BEFORE SAMPLING corresponding segments of the point cloud
	shift = tf.cast(tf.constant([shift_scale*tf.random.normal([1]).numpy()[0], shift_scale*tf.random.normal([1]).numpy()[0], 0.2*shift_scale*tf.random.normal([1]).numpy()[0], 0, 0, 0]), tf.float32)

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 2, draw = False, group = 2, 
		RM = False, DNN_filter = False, cheat = OXTS_ground_truth+shift)

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
		# rand = tf.constant([1., 1., 0.1])*tf.random.normal([ncells, 3]) #larger initial offset
		rand = tf.constant([0.1, 0.1, 0.01])*tf.random.normal([ncells, 3]) #much tighter initial offset

		#only apply rand to compact directions ~~~~~~~~~~~~~~
		rand = it.L @ tf.transpose(it.U, [0,2,1]) @ rand[:,:,None]
		rand = tf.squeeze(rand)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		#tile and apply to scan2
		t = tf.tile(rand, [npts,1])
		t = tf.reshape(tf.transpose(t), [3,npts,-1])
		t = tf.transpose(t, [2,1,0])
		t = tf.reshape(t, [-1, 3])
		scan2 += t.numpy()

		full_soln_vec = rand + shift[:3]
		# compact_soln_vec = it.L @ tf.transpose(it.U, [0,2,1]) @ full_soln_vec[:,:,None] #remove extended axis
		# compact_soln_vec = tf.matmul(it.U, compact_soln_vec) #project back to XYZ
		# compact_soln_vec = compact_soln_vec[:,:,0] #get rid of extra dimension

		soln = full_soln_vec #consider entire solution vector (compact and extended directions)
		# soln = compact_soln_vec #only consider ground truth solutions in directions deemed useful by ICET

		# #Center scan 1 at origin ~~~~~~~~~~~~~~~~~~~~~~~~~
		# # print(scan1)
		# # print(np.shape(scan1))
		# temp1 = np.reshape(scan1, [-1,25,3])
		# # print(np.shape(temp))
		# means = np.reshape(np.mean(temp1, axis = 1), [-1,1,3])
		# # print(np.shape(means))
		# scan1 = temp1-means
		# scan1 = np.reshape(scan1, [-1,3])
		# # print(np.shape(scan1))

		# temp2 = np.reshape(scan2, [-1,25,3])
		# scan2 = temp2 - means
		# scan2 = np.reshape(scan2, [-1,3]) 
		# # print(np.shape(scan2))
		# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		#initialize array to store data on first iteration
		if idx*(j+1) == 0:
			scan1_cum = scan1
			scan2_cum = scan2
			soln_cum = rand + shift[:3]
		else:
			scan1_cum = np.append(scan1_cum, scan1, axis = 0)
			scan2_cum = np.append(scan2_cum, scan2, axis = 0)
			soln_cum = np.append(soln_cum, soln, axis = 0)

	print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

#smol
# np.savetxt('perspective_shift/training_data/ICET_KITTI_FULL_scan1_to10.txt', scan1_cum)
# np.savetxt('perspective_shift/training_data/ICET_KITTI_FULL_scan2_to10.txt', scan2_cum)
# np.savetxt('perspective_shift/training_data/ICET_KITTI_FULL_ground_truth_to10.txt', soln_cum)
# #big
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_scan1_to400.txt', scan1_cum)
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_scan2_to400.txt', scan2_cum)
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_ground_truth_to400.txt', soln_cum)

#direct to npy
np.save("C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_scan1_to400", scan1_cum)
np.save("C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_scan2_to400", scan2_cum)
np.save("C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_ground_truth_to400", soln_cum)
