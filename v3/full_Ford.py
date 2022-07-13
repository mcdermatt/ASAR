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


num_frames = 500 #199

ICET_estimates = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])

for i in range(num_frames):

	print("\n ~~~~~~~~~~~~~~~~~~ Epoch ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	# #partial dataset ---
	# fn1 = 'E:/Ford/IJRR-Dataset-1-subset/SCANS/Scan%04d.mat' %(i+1000)
	# fn2 = 'E:/Ford/IJRR-Dataset-1-subset/SCANS/Scan%04d.mat' %(i+1001)
	# #-------------------

	#full dataset ------
	fn1 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(i+1055) # 980+75
	fn2 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(i+1056) # 980+76
	#NOTE: the good section starts at img 980
	#-------------------

	dat1 = mat4py.loadmat(fn1)
	SCAN1 = dat1['SCAN']
	c1 = np.transpose(np.array(SCAN1['XYZ']))

	dat2 = mat4py.loadmat(fn2)
	SCAN2 = dat2['SCAN']
	c2 = np.transpose(np.array(SCAN2['XYZ']))

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 15, draw = False, group = 2, 
		RM = True, DNN_filter = False, x0 = initial_guess)

	ICET_pred_stds[i] = it.pred_stds
	ICET_estimates[i] = it.X
	# initial_guess = it.X


np.savetxt("Ford_full_pred_stds_v3.txt", ICET_pred_stds)
np.savetxt("Ford_full_estimates_v3.txt", ICET_estimates)

#OLD (using partial Ford dataset)
#v1 - fid 50, dnn = 0.10, moving = 0.1
#v2 - fid 90, dnn = 0.05, moving = 0.1
#v3 - fid 50, no dnn or RM
#v4 - pure dnn, fid = 70, niter = 20
#v5 - pure dnn, fid = 50, niter = 15

#New (using complete Ford dataset)
#v1 - fid 50, no dnn, moving = ??, 25 frame test
#v2 - first 300 frames
#v3 - 300 frames starting with frame corresponding to img980