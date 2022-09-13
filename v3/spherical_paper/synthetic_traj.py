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
nframes = 40 #42 #how many sequential frames to consider (num pairs = num frames - 1)
niter = 3   #number of times to repeat each pair of scans


ICET_estimates = np.zeros([niter*(nframes - 1), 6])
ICET_pred_stds = np.zeros([niter*(nframes - 1), 6])

for idx in range(1, nframes):
	for i in range(niter):
		print("------------- Epoch ", (idx-1)*niter + i, "---------------")

		fn1 = "MC_trajectories/scene" + str(scene) + "_scan" + str(idx) + ".txt" 
		fn2 = "MC_trajectories/scene" + str(scene) + "_scan" + str(idx + 1) + ".txt" 

		c1_raw = np.loadtxt(fn1, dtype = float) #Scene 1
		c2_raw = np.loadtxt(fn2, dtype = float)

		# c1_raw = np.loadtxt("mountain_scan1_no_trees.txt", dtype = float) #Scene 2
		# c2_raw = np.loadtxt("mountain_scan2_no_trees.txt", dtype = float)

		#add noise (if not generated when point clouds were created)
		c1 = c1_raw + 0.02*np.random.randn(np.shape(c1_raw)[0], 3)
		c2 = c2_raw + 0.02*np.random.randn(np.shape(c2_raw)[0], 3) 
		rot = R_tf(tf.constant([0., 0., 0.05]))
		c2 = c2 @ rot.numpy() 


		## normal ICET-----------------------------------------------------
		# it = ICET(cloud1 = c1, cloud2 = c2, fid = 90, niter = 25, 
		# 	draw = False, group = 2, RM = True, DNN_filter = False)
		#------------------------------------------------------------------

		#using naive spherical cuboid-shaped voxles------------------------
		x0 = tf.constant([0.5 + 0.1*np.random.randn(), 0, 0, 0, 0, 0])
		c1 = c1[c1[:,2] > -1.55] #ignore ground plane
		c2 = c2[c2[:,2] > -1.55] #ignore ground plane
		it = ICET(cloud1 = c1, cloud2 = c2, fid = 40, niter = 20, 
			draw = False, group = 1, RM = False, DNN_filter = False, x0 = x0)#, cheat = gt)
		#------------------------------------------------------------------


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



# np.save("MC_results/traj1_spherical_ICET_estimates_test", ICET_estimates)
# np.save("MC_results/traj1_spherical_ICET_pred_stds_test", ICET_pred_stds)

# np.save("MC_results/traj1_spherical_NDT_estimates_v2", ICET_estimates)
np.save("MC_results/traj1_naive_NDT_estimates_v2", ICET_estimates)


# np.save("MC_results/traj2_spherical_ICET_estimates", ICET_estimates)
# np.save("MC_results/traj2_spherical_ICET_pred_stds", ICET_pred_stds)
