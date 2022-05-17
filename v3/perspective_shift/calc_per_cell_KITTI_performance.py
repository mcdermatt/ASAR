from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import numpy as np
import tensorflow as tf
import time

#load model
# model = tf.keras.models.load_model("Net.kmod")
# model = tf.keras.models.load_model("PIPEnet10.kmod")
# model = tf.keras.models.load_model("KITTInet.kmod")
# model = tf.keras.models.load_model("KITTInet50.kmod")
model = tf.keras.models.load_model("FORDnet.kmod")

#read in dense point cloud of car
points_per_sample = 25 #num pts per scan - defined in MatLab script
d1 = np.loadtxt('training_data/ICET_KITTI_scan1.txt')
d2 = np.loadtxt('training_data/ICET_KITTI_scan2.txt')
gt = np.loadtxt('training_data/ICET_KITTI_ground_truth.txt')

# points_per_sample = 50
# d1 = np.loadtxt('training_data/ICET_KITTI_scan1_50.txt')
# d2 = np.loadtxt('training_data/ICET_KITTI_scan2_50.txt')
# gt = np.loadtxt('training_data/ICET_KITTI_ground_truth_50.txt')
scan1 = tf.reshape(tf.convert_to_tensor(d1), [-1, points_per_sample, 3])
scan2 = tf.reshape(tf.convert_to_tensor(d2), [-1, points_per_sample, 3])

#split data into training and validation sets
tsplit = 0.95 #this fraction goes into training
ntrain = int(tsplit*tf.shape(scan1)[0].numpy())
x_train = tf.concat((scan1[:ntrain], scan2[:ntrain]), axis = 1)
x_test = tf.concat((scan1[ntrain:], scan2[ntrain:]), axis = 1)
y_test = gt[ntrain:]

niter = 90
err  = np.zeros([niter, 3])
for j in range(niter):
	print(j)
	n = j
	#appy model to points
	c1 = np.array([x_test[n,:points_per_sample,0].numpy(), x_test[n,:points_per_sample,1].numpy(), x_test[n,:points_per_sample,2].numpy()])
	c2 = np.array([x_test[n,points_per_sample:,0].numpy(), x_test[n,points_per_sample:,1].numpy(), x_test[n,points_per_sample:,2].numpy()])

	inputs = x_test[n][None,:]
	runlen = 10 #number of iterations to run iterative PC matching
	correction = 0
	for i in range(runlen):
	    correction += model.predict(inputs)[0] #show what the network thinks
	#     correction = 0.1*y_test[n] #show actual solution
	    c1_new = np.array([c1[0,:] + correction[0], c1[1,:] + correction[1], c1[2,:] + correction[2]])
	    inputs = np.append(c1_new, c2, axis = 1).T[None,:,:]

	err[j] = correction - y_test[n]

print("err: \n", err)
print("mean err: \n", np.mean(err, axis = 0))
print("std err: \n", np.std(err, axis = 0))