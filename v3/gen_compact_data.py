import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf
import mat4py

numShifts = 20 #number of times to resample and translate each voxel each scan
runLen = 200 #199
ptsPerCell = 50
start_idx = 850 #2700 #1050 #2150

ground_truth = np.loadtxt("E:/Ford/IJRR-Dataset-1/SCANS/truth.txt")/10
ground_truth = tf.cast(tf.convert_to_tensor(ground_truth), tf.float32)

print("\n gt", tf.shape(ground_truth))

for idx in range(runLen):
	print("\n ~~~~~~~~~ Frame #", idx, "~~~~~~~~~~~~~ \n")

	fn1 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(idx+start_idx + 75) #TEST add 1 to get spacing correct??
	fn2 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(idx+start_idx + 76)

	dat1 = mat4py.loadmat(fn1)
	SCAN1 = dat1['SCAN']
	c1 = np.transpose(np.array(SCAN1['XYZ']))

	dat2 = mat4py.loadmat(fn2)
	SCAN2 = dat2['SCAN']
	c2 = np.transpose(np.array(SCAN2['XYZ']))
	c1 = c1[c1[:,2] > -2.2] #ignore ground plane
	c2 = c2[c2[:,2] > -2.2] #ignore ground plane

	gt = (ground_truth[idx+start_idx - 2,:] + ground_truth[idx + start_idx - 1,:])/2

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 3, draw = False, group = 2, 
		RM = True, DNN_filter = False, cheat = gt)

	#Get ragged tensor containing all points from each scan inside each sufficient voxel
	in1 = it.inside1
	npts1 = it.npts1
	in2 = it.inside2
	npts2 = it.npts2
	corr = it.corr #indices of bins that have enough points from scan1 and scan2

	#get indices of rag with >= 25 elements
	ncells = tf.shape(corr)[0].numpy() #num of voxels with sufficent number of points
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

		#NEW: suppress extended axis -----------------------------------
		# print("OG U", tf.shape(it.U))

		LUT = tf.matmul(it.L, tf.transpose(it.U, [0,2,1]))
		# print("-- LUT", tf.shape(LUT))

		# dz_new = tf.matmul(LUT, dnnsoln[:,:,None])
		rand_compact = tf.matmul(LUT, rand[:,:,None])
		rand_compact_xyz = tf.matmul(it.U, rand_compact)

		# print("\n LUT", tf.shape(LUT))
		#---------------------------------------------------------------

		# print("before \n", np.shape(it.corn.numpy()))
		# print("after \n", it.s2c(it.corn.numpy()))

		if idx*(j+1) == 0:
			scan1_cum = scan1
			scan2_cum = scan2
			# rand_cum = rand #old - use for D2D vs DNN notebook
			rand_cum = rand_compact_xyz #only compact
			LUT_cum = LUT.numpy()
			L_cum = it.L.numpy()
			U_cum = it.U.numpy()
			corn_cum = it.corn.numpy()
			# corn_cum = it.s2c(it.corn.numpy()) #test
		else:
			scan1_cum = np.append(scan1_cum, scan1, axis = 0)
			scan2_cum = np.append(scan2_cum, scan2, axis = 0)
			# rand_cum = np.append(rand_cum, rand, axis = 0) #old - use for D2D vs DNN notebook
			rand_cum = np.append(rand_cum, rand_compact_xyz, axis = 0) #overwrites soln with only compact component
			LUT_cum = np.append(LUT_cum, LUT.numpy(), axis = 0)
			U_cum = np.append(U_cum, it.U.numpy(), axis = 0)
			L_cum = np.append(L_cum, it.L.numpy(), axis = 0)
			corn_cum = np.append(corn_cum, it.corn.numpy(), axis = 0)
			# corn_cum = np.append(corn_cum, it.s2c(it.corn.numpy()), axis = 0) #test


	print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

#small files
np.save('perspective_shift/training_data/compact_scan1', scan1_cum)
np.save('perspective_shift/training_data/compact_scan2', scan2_cum)
np.save('perspective_shift/training_data/compact_ground_truth', rand_cum)
np.save('perspective_shift/training_data/LUT', LUT_cum) 
np.save('perspective_shift/training_data/L', L_cum)
np.save('perspective_shift/training_data/U', U_cum)
np.save('perspective_shift/training_data/corn', corn_cum)
# TODO - save ground truth SEPRATE from compact ground truth

#big files
# np.save('C:/Users/Derm/Desktop/big/pshift/compact_scan1', scan1_cum)
# np.save('C:/Users/Derm/Desktop/big/pshift/compact_scan2', scan2_cum)
# np.save('C:/Users/Derm/Desktop/big/pshift/compact_ground_truth', rand_cum)
# np.save('C:/Users/Derm/Desktop/big/pshift/LUT', LUT_cum)
# np.save('C:/Users/Derm/Desktop/big/pshift/L', L_cum)
# np.save('C:/Users/Derm/Desktop/big/pshift/U', U_cum)
# np.save('C:/Users/Derm/Desktop/big/pshift/corn', corn_cum)
# #TODO - save ground truth SEPRATE from compact ground truth
