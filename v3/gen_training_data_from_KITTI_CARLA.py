"""generates data for training perspective shift filter DNN from KITTI sample dataset"""

import pykitti
import numpy as np
import tensorflow as tf
import trimesh

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
from utils import R_tf
from metpy.calc import lat_lon_grid_deltas

numShifts = 10 #5 #number of times to resample and translate each voxel each scan
noise_scale = 0.005 #add gaussian noise to each point
start_idx = 700 #1150 for town 02
runLen = 350
npts = 100 #50 

fpl = np.loadtxt("/home/derm/KITTICARLA/dataset/Town01/generated/full_poses_lidar.txt") #full poses lidar

#create rotation and translation vectors
R = np.array([[fpl[:,0], fpl[:,1], fpl[:,2]],
              [fpl[:,4], fpl[:,5], fpl[:,6]],
              [fpl[:,8], fpl[:,9], fpl[:,10]]]).T

T = np.array([fpl[:,3], fpl[:,7], fpl[:,11]]).T
vel = np.diff(T.T)

for idx in range(runLen):
	
	skip = int(3*np.random.randn())

	s1_fn = '/home/derm/KITTICARLA/dataset/Town01/generated/frames/frame_%04d.ply' %(start_idx + idx)
	s2_fn = '/home/derm/KITTICARLA/dataset/Town01/generated/frames/frame_%04d.ply' %(start_idx + idx + skip)

	dat1 = trimesh.load(s1_fn)
	dat2 = trimesh.load(s2_fn)

	c1 = dat1.vertices
	c1 = c1[c1[:,2] > -1.65]
	c1 = c1.dot(R[(idx)*100])
	c1 += noise_scale*np.random.randn(np.shape(c1)[0],3)

	c2 = dat2.vertices
	c2 = c2[c2[:,2] > -1.65]
	c2 = c2.dot(R[(idx+skip)*100])
	# c2 = c2 + vel[:,(idx+skip)*100]*100*skip #transform c2 to overlay with c1
	c2 = c2 + (vel[:,(idx+skip)*100] + vel[:,(idx)*100])*50*skip #transform c2 to overlay with c1
	c2 += noise_scale*np.random.randn(np.shape(c2)[0],3)

	shift_scale = 0.05 #standard deviation by which to shift the grid BEFORE SAMPLING corresponding segments of the point cloud
	shift = tf.cast(tf.constant([shift_scale*tf.random.normal([1]).numpy()[0], shift_scale*tf.random.normal([1]).numpy()[0], 0.2*shift_scale*tf.random.normal([1]).numpy()[0], 0, 0, 0]), tf.float32)

	x0 = tf.constant([0., 0., 0., 0., 0., 0.])
	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 2, draw = False, group = 2, 
		RM = False, DNN_filter = False, cheat = x0+shift)

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
			soln_cum = soln
		else:
			scan1_cum = np.append(scan1_cum, scan1, axis = 0)
			scan2_cum = np.append(scan2_cum, scan2, axis = 0)
			soln_cum = np.append(soln_cum, soln, axis = 0)

	print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

np.save('/media/derm/06EF-127D1/TrainingData/KITTI_CARLA_01_scan1_100pts', scan1_cum)
np.save('/media/derm/06EF-127D1/TrainingData/KITTI_CARLA_01_scan2_100pts', scan2_cum)
np.save('/media/derm/06EF-127D1/TrainingData/KITTI_CARLA_01_ground_truth_100pts', soln_cum)