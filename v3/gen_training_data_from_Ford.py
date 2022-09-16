import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf
import mat4py

#NOTE: after looking at results of multiple differnet point cloud registration algorithms on the Ford Dataset, 
#		it is clear that there are obvious errors in the "ground truth" position. 
#		Frames 650-790 seem to be at least within 1cm of true states (too simple)
#		1320-1430 also good (more complex terrain)

numShifts = 5 #20 #number of times to resample and translate each voxel each scan
runLen = 10 #300
ptsPerCell = 100
start_idx = 1130 #1310

ground_truth = np.loadtxt("E:/Ford/IJRR-Dataset-1/SCANS/truth.txt")/10 #was this
# ground_truth = np.loadtxt("E:/Ford/IJRR-Dataset-1/SCANS/truth_test.txt")/10 #truth_test has forward and lateral axis of translation flipped
#flip axis
# temp = ground_truth[:,:]
# temp[:,0] = ground_truth[:,1]
# temp[:,1] = ground_truth[:,0]
# temp[:,2] = -temp[:,2] #flip z axis
# ground_truth[:,:] = temp[:,:]
ground_truth = tf.cast(tf.convert_to_tensor(ground_truth), tf.float32)

print("\n gt", tf.shape(ground_truth))

for idx in range(runLen):
	print("\n ~~~~~~~~~ Frame #", idx, "~~~~~~~~~~~~~ \n")

	fn1 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(idx+start_idx + 75)
	# fn2 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(idx+start_idx + 76) #was this
	fn2 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(idx+start_idx + 78) #test


	dat1 = mat4py.loadmat(fn1)
	SCAN1 = dat1['SCAN']
	c1 = np.transpose(np.array(SCAN1['XYZ']))

	dat2 = mat4py.loadmat(fn2)
	SCAN2 = dat2['SCAN']
	c2 = np.transpose(np.array(SCAN2['XYZ']))

	c1 = c1[c1[:,2] > -2.2] #ignore ground plane
	c2 = c2[c2[:,2] > -2.2] #ignore ground plane

	# gt = (ground_truth[idx+start_idx - 2,:] + ground_truth[idx + start_idx - 1,:])/2 #for transformation between subsequent frames

	#test
	gt1 = (ground_truth[idx+start_idx - 2,:] + ground_truth[idx + start_idx - 1,:])/2
	gt2 = (ground_truth[idx+start_idx - 1,:] + ground_truth[idx + start_idx,:])/2
	gt3 = (ground_truth[idx+start_idx,:] + ground_truth[idx + start_idx + 1,:])/2
	gt = gt1 + gt2 + gt3


	# it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 3, draw = False, group = 2, 
	# 	RM = True, DNN_filter = False, cheat = gt)
	#test
	it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 3, draw = False, group = 2, 
		RM = False, DNN_filter = False, cheat = gt)

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
			# rand_cum = rand + gt[:3] #wrong
			rand_cum = rand #new 8/3
		else:
			scan1_cum = np.append(scan1_cum, scan1, axis = 0)
			scan2_cum = np.append(scan2_cum, scan2, axis = 0)
			# rand_cum = np.append(rand_cum, rand + gt[:3], axis = 0) #wrong
			rand_cum = np.append(rand_cum, rand, axis = 0)  #new 8/3
 

	print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

#smol
# np.savetxt('perspective_shift/training_data/ICET_Ford_scan1.txt', scan1_cum)
# np.savetxt('perspective_shift/training_data/ICET_Ford_scan2.txt', scan2_cum)
# np.savetxt('perspective_shift/training_data/ICET_Ford_ground_truth.txt', rand_cum)

#big
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_Ford_v2_scan1.txt', scan1_cum)
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_Ford_v2_scan2.txt', scan2_cum)
# np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_Ford_v2_ground_truth.txt', rand_cum)
np.save('D:/TrainingData/Ford_scan1_100pts', scan1_cum)
np.save('D:/TrainingData/Ford_scan2_100pts', scan2_cum)
np.save('D:/TrainingData/Ford_ground_truth_100pts', rand_cum)
# np.save('D:/TrainingData/Ford_scan1_100pts_large_displacement', scan1_cum)
# np.save('D:/TrainingData/Ford_scan2_100pts_large_displacement', scan2_cum)
# np.save('D:/TrainingData/Ford_ground_truth_100pts_large_displacement', rand_cum)

#v1 = 1050
#v2 = 2150
#v3 = 2700
#v4 = 850

#for tf.data test
# np.save('C:/Users/Derm/Desktop/big/pshift/test1_scan1', scan1_cum)
# np.save('C:/Users/Derm/Desktop/big/pshift/test1_scan2', scan2_cum)
# np.save('C:/Users/Derm/Desktop/big/pshift/test1_ground_truth', rand_cum)
