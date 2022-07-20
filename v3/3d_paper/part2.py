from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from ICET_spherical import ICET
from utils import R_tf

""" Script to generate Monte-Carlo results for experiment 2 of 3D-ICET paper """

#NOTE: to remove extended axis removal operation, un-comment out <axislen_actual = 0.1*tf.math....> in get_U_and_L_cluster()>

niter = 500 #10

c1_raw = np.loadtxt("big_curve_scan1.txt", dtype = float)
# c2_raw = np.loadtxt("big_curve_scan2.txt", dtype = float)
c2_raw = np.loadtxt("big_curve_scan1.txt", dtype = float) #use same scan as first frame


ICET_estimates = np.zeros([niter, 6])
ICET_pred_stds = np.zeros([niter, 6])

for i in range(niter):
	print("------------- Epoch ", i, "---------------")

	# #add noise (if not generated when point clouds were created)
	noise_scale = 0.02
	c1 = c1_raw + noise_scale*np.random.randn(np.shape(c1_raw)[0], 3)
	c2 = c2_raw + noise_scale*np.random.randn(np.shape(c2_raw)[0], 3) 

	# #slightly raise each PC
	# c1[:,2] += 0.2
	# c2[:,2] += 0.2

	#translate scan2 
	random_translation = 0.125*tf.constant([np.random.randn(), np.random.randn(), 0.0*np.random.randn()])
	c2 += random_translation

	#rotate scans
	# rot = R_tf(tf.constant([0., 0., 0.05]))
	random_rotation = 0.03*tf.constant([0.0*np.random.randn(), 0.0*np.random.randn(), np.random.randn()])
	rot = R_tf(random_rotation)
	c2 = c2 @ rot.numpy() 

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 100, niter = 5, 
		draw = False, group = 2, RM = False, DNN_filter = False)

	# it = ICET(cloud1 = c1, cloud2 = c2, fid = 100, niter = 20, 
	# 	draw = False, group = 2, RM = False, DNN_filter = False, x0 = it.X)

	ICET_estimates[i] = it.X
	
	#test 
	ICET_estimates[i,:3] += random_translation
	ICET_estimates[i,3:] += random_rotation
	
	ICET_pred_stds[i] = it.pred_stds

# #with mitigation
# np.save("MC_results/scene2_ICET_estimates_v3", ICET_estimates)
# np.save("MC_results/scene2_ICET_pred_stds_v3", ICET_pred_stds)

#no mitigation
np.save("MC_results/scene2_ICET_estimates_NM_v3", ICET_estimates)
np.save("MC_results/scene2_ICET_pred_stds_NM_v3", ICET_pred_stds)
