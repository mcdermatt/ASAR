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


num_frames = 3810 #500 #199

ICET_estimates = np.zeros([num_frames, 6])
ICET_pred_stds = np.zeros([num_frames, 6])

initial_guess = tf.constant([0., 0., 0., 0., 0., 0.])

gt_fn = "spherical_paper/ford_results/truth_body_frame.txt";
gt = np.loadtxt(gt_fn);

for i in range(num_frames):

	print("\n ~~~~~~~~~~~~~~~~~~ Epoch ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	# #partial dataset ---
	# fn1 = 'E:/Ford/IJRR-Dataset-1-subset/SCANS/Scan%04d.mat' %(i+1000)
	# fn2 = 'E:/Ford/IJRR-Dataset-1-subset/SCANS/Scan%04d.mat' %(i+1001)
	# #-------------------

	#full dataset ------
	fn1 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(i + 75)
	fn2 = 'E:/Ford/IJRR-Dataset-1/SCANS/Scan%04d.mat' %(i + 76)
	#NOTE: 
	#	campus townhouses at 980+75,76
	#	long straight strip at 2200 + 75,76
	#   v8 @ 1175 ++
	#-------------------

	dat1 = mat4py.loadmat(fn1)
	SCAN1 = dat1['SCAN']
	c1 = np.transpose(np.array(SCAN1['XYZ']))

	dat2 = mat4py.loadmat(fn2)
	SCAN2 = dat2['SCAN']
	c2 = np.transpose(np.array(SCAN2['XYZ']))

	#seed initial registration estimate
	initial_guess = tf.constant([0, 0.1*gt[i,1] + 0.01*np.random.randn(), 0, 0, 0, 0,])

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 15, draw = False, group = 2, 
		RM = True, DNN_filter = False, x0 = initial_guess)

	ICET_pred_stds[i] = it.pred_stds
	ICET_estimates[i] = it.X
	initial_guess = it.X


np.savetxt("Ford_full_pred_stds_v10.txt", ICET_pred_stds)
np.savetxt("Ford_full_estimates_v10.txt", ICET_estimates)

#OLD (using partial Ford dataset)
#v1 - fid 50, dnn = 0.10, moving = 0.1
#v2 - fid 90, dnn = 0.05, moving = 0.1
#v3 - fid 50, no dnn or RM
#v4 - pure dnn, fid = 70, niter = 20
#v5 - pure dnn, fid = 50, niter = 15

#New (using complete Ford dataset)
#v1 - fid 50, no dnn, moving = ??, 25 frame test
#v2 - first 300 frames, NO DNN
#v3 - 300 frames starting with frame corresponding to img980, NO DNN
#V4 - 500 frames in main downtown area, NO DNN
#v5 - every frame, NO DNN
#v6 - FIRST TEST WITH DNN, fid = 50, niter = 10, RM&DNN thresh = 0.05
#v7 - WITH DNN, fid = 70, niter = 10, RM&DNN thresh = 0.02/0.03
#v8 - Better DNN, tighter convergence!! fid = 70, niter = 13, thresh = 0.4, starts at 1175
#v9 - same param as v8, but without DNN
#v10 - full Ford dataset, filting by translation and rotation. 