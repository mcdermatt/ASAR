from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
import mat4py


num_frames = 199

ICET_estimates = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])

for i in range(num_frames):

	fn1 = 'E:/Ford/IJRR-Dataset-1-subset/SCANS/Scan%04d.mat' %(i+1000)
	fn2 = 'E:/Ford/IJRR-Dataset-1-subset/SCANS/Scan%04d.mat' %(i+1001)

	dat1 = mat4py.loadmat(fn1)
	SCAN1 = dat1['SCAN']
	c1 = np.transpose(np.array(SCAN1['XYZ']))

	dat2 = mat4py.loadmat(fn2)
	SCAN2 = dat2['SCAN']
	c2 = np.transpose(np.array(SCAN2['XYZ']))

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 20, draw = False, group = 2, 
		RM = True, DNN_filter = True, x0 = initial_guess)

	ICET_pred_stds[i] = it.pred_stds
	ICET_estimates[i] = it.X
	initial_guess = it.X


np.savetxt("Ford_pred_stds_v1.txt", ICET_pred_stds)
np.savetxt("Ford_estimates_v1.txt", ICET_estimates)

#v1 - fid 50, dnn = 0.10, moving = 0.1
#v2 - fid 90, dnn = 0.05, moving = 0.1