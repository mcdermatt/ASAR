import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf
import h5py


numShifts = 10
runLen = 124 #max 125
ptsPerCell = 50

# filename = 'C:/CODD/data/m1v7p7s769.hdf5' #straight line urban
# filename = 'C:/CODD/data/m10v11p6s30.hdf5' #wide road, palm trees, and traffic
filename = 'C:/CODD/data/m2v7p3s333.hdf5' #complex geometries


vidx = 1 #set vehicle index
with h5py.File(filename, 'r') as hf:
#     pcls = hf['point_cloud'][:]
    #[frames, vehicles, points_per_cloud, 4]
    pcls = hf['point_cloud'][:, vidx ,: , :3]
    #[frames, points_per_cloud, rgb]
#     pose = hf['lidar_pose'][:]
    #[frames, vehicles, (x,y,z, rotx, roty, rotz)]
    pose = hf['lidar_pose'][:, vidx, :]    
    vbb = hf['vehicle_boundingbox'][:, vidx, :]

vel = np.diff(pose, axis = 0)
vf = np.sqrt(vel[:,0]**2 + vel[:,1]**2 )

#convert ground truth yaw from deg2rad
vel[:, 3:] = np.deg2rad(vel[:,3:])

for idx in range(runLen):
	print("\n ~~~~~~~~~ Frame #", idx, "~~~~~~~~~~~~~ \n")

	c1 = pcls[idx]
	c2 = pcls[idx+1]

	noise_scale = 0.005#0.005 #0.01 # doesn't work at 0.001
	c1 += noise_scale*np.random.randn(np.shape(c1)[0], 3)
	c2 += noise_scale*np.random.randn(np.shape(c2)[0], 3)

	c1 = c1[c1[:,2] > -2.2] #ignore ground plane
	c2 = c2[c2[:,2] > -2.2] #ignore ground plane

	gt = tf.constant([vf[idx], 0., 0., 0., 0., vel[idx, 4]])

	# it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 3, draw = False, group = 2, 
	# 	RM = True, DNN_filter = False, cheat = gt)

	shift_scale = 0.0 #standard deviation by which to shift the grid BEFORE SAMPLING corresponding segments of the point cloud
	shift = tf.cast(tf.constant([shift_scale*tf.random.normal([1]).numpy()[0], shift_scale*tf.random.normal([1]).numpy()[0], 0.2*shift_scale*tf.random.normal([1]).numpy()[0], 0, 0, 0]), tf.float32)
	#test
	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 1, draw = False, group = 2, 
		RM = False, DNN_filter = False, cheat = gt+shift)


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
		idx1 = np.zeros([ncells ,ptsPerCell])
		idx2 = np.zeros([ncells ,ptsPerCell])

		#loop through each element of ragged tensor
		for i in range(ncells):
		    idx1[i,:] = tf.random.shuffle(enough1[i])[:ptsPerCell].numpy() #shuffle order and take first N elements
		    idx2[i,:] = tf.random.shuffle(enough2[i])[:ptsPerCell].numpy() #shuffle order and take first N elements

		idx1 = tf.cast(tf.convert_to_tensor(idx1), tf.int32) #indices in scan 1 of points we've selected
		idx2 = tf.cast(tf.convert_to_tensor(idx2), tf.int32) 

		from1 = tf.gather(it.cloud1_tensor, idx1)
		# from2 = tf.gather(it.cloud2_tensor_OG, idx2) #corresponding points in OG pose
		from2 = tf.gather(it.cloud2_tensor, idx2) 	   #transformed by "ground truth" translation

		scan1 = tf.reshape(from1, [-1, 3]).numpy()
		scan2 = tf.reshape(from2, [-1, 3]).numpy()

		#randomly translate each sample from scan 2
		rand = tf.constant([1., 1., 0.1])*tf.random.normal([ncells, 3])
		# rand = tf.constant([0.001, 0.001, 0.001])*tf.random.normal([ncells, 3]) #test
		#tile and apply to scan2
		t = tf.tile(rand, [ptsPerCell,1])
		t = tf.reshape(tf.transpose(t), [3,ptsPerCell,-1])
		t = tf.transpose(t, [2,1,0])
		t = tf.reshape(t, [-1, 3])
		scan2 += t.numpy()

		#randomly flip scan1 and scan2 -----------------
		#doing this to prevent systemic bias in perspective shift form forward motion
		if np.random.randn() > 0:
			temp = scan2.copy()
			scan2 = scan1
			scan1 = temp
			rand = -rand
		#-----------------------------------------------

		if idx*(j+1) == 0:
			scan1_cum = scan1
			scan2_cum = scan2
			rand_cum = rand - shift[:3]
		else:
			scan1_cum = np.append(scan1_cum, scan1, axis = 0)
			scan2_cum = np.append(scan2_cum, scan2, axis = 0)
			rand_cum = np.append(rand_cum, rand - shift[:3], axis = 0)  #new 8/3
 

	print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

np.save('D:/TrainingData/CODD_v2_scan1_50pts', scan1_cum)
np.save('D:/TrainingData/CODD_v2_scan2_50pts', scan2_cum)
np.save('D:/TrainingData/CODD_v2_ground_truth_50pts', rand_cum)