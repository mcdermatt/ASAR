import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf
import mat4py

#TODO- should I be generating training data without the ground plane???

numShifts = 5 #number of times to resample and translate each voxel each scan
runLen = 350 #199
ptsPerCell = 50

# ground_truth = np.loadtxt("E:/Ford/IJRR-Dataset-1-subset/SCANS/truth.txt")/10
# ground_truth = tf.cast(tf.convert_to_tensor(ground_truth), tf.float32)


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

	# fn1 = 'E:/Ford/IJRR-Dataset-1-subset/SCANS/Scan%04d.mat' %(idx+1000)
	# fn2 = 'E:/Ford/IJRR-Dataset-1-subset/SCANS/Scan%04d.mat' %(idx+1001)

	fn1 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(idx+1150 + 75) #TEST add 1 to get spacing correct??
	fn2 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(idx+1151 + 75)

	dat1 = mat4py.loadmat(fn1)
	SCAN1 = dat1['SCAN']
	c1 = np.transpose(np.array(SCAN1['XYZ']))

	dat2 = mat4py.loadmat(fn2)
	SCAN2 = dat2['SCAN']
	c2 = np.transpose(np.array(SCAN2['XYZ']))

	c1 = c1[c1[:,2] > -2.2] #ignore ground plane
	c2 = c2[c2[:,2] > -2.2] #ignore ground plane

	gt = (ground_truth[idx+1150,:] + ground_truth[idx+1151,:])/2 #avg between pts

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 3, draw = False, group = 2, 
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
np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_Ford_scan1.txt', scan1_cum)
np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_Ford_scan2.txt', scan2_cum)
np.savetxt('C:/Users/Derm/Desktop/big/pshift/ICET_Ford_ground_truth.txt', rand_cum)
