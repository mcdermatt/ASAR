from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from ICET_spherical import ICET
from utils import R_tf

""" Script to generate transformation estimates FROM ENTIRE TRAJECTORIES
 (Spherical ICET Paper) """

scene = 1   #scene 1 = highway, scene 2 = mountain
nframes = 10 #42 #how many sequential frames to consider (num pairs = num frames - 1)
niter = 1   #number of times to repeat each pair of scans


ICET_estimates = np.zeros([niter*(nframes - 1), 6])
ICET_pred_stds = np.zeros([niter*(nframes - 1), 6])

for idx in range(1, nframes):
	for i in range(niter):
		print("------------- Epoch ", (idx-1)*niter + i, "---------------")

		fn1 = "figures/MC_trajectories/forest1_scan" + str(idx) + ".txt" 
		fn2 = "figures/MC_trajectories/forest1_scan" + str(idx + 1) + ".txt" 

		c1_raw = np.loadtxt(fn1, dtype = float) #Scene 1
		c2_raw = np.loadtxt(fn2, dtype = float)

		# c1_raw = np.loadtxt("mountain_scan1_no_trees.txt", dtype = float) #Scene 2
		# c2_raw = np.loadtxt("mountain_scan2_no_trees.txt", dtype = float)

		#add noise (if not generated when point clouds were created)
		c1 = c1_raw + 0.02*np.random.randn(np.shape(c1_raw)[0], 3)
		c2 = c2_raw + 0.02*np.random.randn(np.shape(c2_raw)[0], 3) 
		# rot = R_tf(tf.constant([0., 0., 0.05]))
		# c2 = c2 @ rot.numpy() 

		# c1 = c1[c1[:,2] > -2.] #ignore ground plane
		# c2 = c2[c2[:,2] > -2.] #ignore ground plane

		x0 = tf.constant([3.5, 0, 0, 0, 0, 0])

		it = ICET(cloud1 = c1, cloud2 = c2, fid = 90, niter = 20, 
			draw = False, group = 2, RM = True, DNN_filter = True, x0 = x0)
	
		ICET_estimates[(idx-1)*niter + i] = it.X
		ICET_pred_stds[(idx-1)*niter + i] = it.pred_stds


		# #to save unshadowed points to file for benchmarking with ICP--------

		# np.savetxt("unshadowed_points/scene_1_scan_" + str(idx) + "A_no_shadows.txt", it.cloud1_static)
		# np.savetxt("unshadowed_points/scene_1_scan_" + str(idx+1) + "B_no_shadows.txt", it.cloud2_static)

		# #TODO: save each frame A and B for when it is the first scan and when it is the 2nd scan
		# #     if we were to use our anti-shadowing grid + ICP we would remove shadows from BOTH scans
		# #	  before registering with ICP

		# #ex: use frames 5A and 6B together

		# #Still need to transform points from it.cloud2_static back to origonal positions??

		# #-------------------------------------------------------------------

np.save("figures/MC_results/forest1_NDT_withDNN", ICET_estimates)
