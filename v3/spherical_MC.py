from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf

""" Script to generate Monte-Carlo results for spherical ICET (Spherical Coordinates Paper) """

niter = 100

c1_raw = np.loadtxt("scene1_scan1.txt", dtype = float) #Scene 1
c2_raw = np.loadtxt("scene1_scan2.txt", dtype = float)

# c1_raw = np.loadtxt("mountain_scan1_no_trees.txt", dtype = float) #Scene 2
# c2_raw = np.loadtxt("mountain_scan2_no_trees.txt", dtype = float)


ICET_estimates = np.zeros([niter, 6])
ICET_pred_stds = np.zeros([niter, 6])

for i in range(niter):
	print("------------- Epoch ", i, "---------------")

	#add noise (if not generated when point clouds were created)
	c1 = c1_raw + 0.02*np.random.randn(np.shape(c1_raw)[0], 3)
	c2 = c2_raw + 0.02*np.random.randn(np.shape(c2_raw)[0], 3) 

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 10, 
		draw = False, group = 2, RM = False, DNN_filter = False)

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 100, niter = 20, 
		draw = False, group = 2, RM = False, DNN_filter = False, x0 = it.X)

	ICET_estimates[i] = it.X
	ICET_pred_stds[i] = it.pred_stds

np.save("MC_results/s1_spherical_ICET_estimates", ICET_estimates)
np.save("MC_results/s1_spherical_ICET_pred_stds", ICET_pred_stds)

# np.save("MC_results/s2_spherical_ICET_estimates_no_trees_with_rotation", ICET_estimates)
# np.save("MC_results/s2_spherical_ICET_pred_stds_no_trees_with_rotation", ICET_pred_stds)