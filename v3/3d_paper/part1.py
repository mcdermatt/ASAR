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

#limit GPU memory ------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    memlim = 12*1024
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memlim)])
  except RuntimeError as e:
    print(e)
#-----------------------------------------------------------------


""" Script to generate Monte-Carlo results for experiment 1 of 3D-ICET paper """

#NOTE: to remove extended axis removal operation, un-comment out <axislen_actual = 0.1*tf.math....> in get_U_and_L_cluster()>

niter = 50

#scene 1
# c1_raw = np.loadtxt("T_intersection_scan1.txt", dtype = float)
# # c2_raw = np.loadtxt("T_intersection_scan2.txt", dtype = float)
# c2_raw = np.loadtxt("T_intersection_scan1.txt", dtype = float) #generate both scans from same location (remove shadowing bias)

c1_raw = np.loadtxt("T_intersection_simple_scan1.txt", dtype = float)
# c2_raw = np.loadtxt("T_intersection_simple_scan2.txt", dtype = float)
c2_raw = np.loadtxt("T_intersection_simple_scan1.txt", dtype = float) #test


ICET_estimates = np.zeros([niter, 6])
ICET_pred_stds = np.zeros([niter, 6])

fail_count = 0

for i in range(niter):
	print("------------- Epoch ", i, "---------------")

	# #add noise (if not generated when point clouds were created)
	# noise_scale = 0.02 #old
	noise_scale = 0.05 
	c1 = c1_raw + noise_scale*np.random.randn(np.shape(c1_raw)[0], 3)
	c2 = c2_raw + noise_scale*np.random.randn(np.shape(c2_raw)[0], 3) 

	# #slightly raise each PC
	# c1[:,2] += 0.2
	# c2[:,2] += 0.2

	#translate scan2 
	# random_translation = 0.125*tf.constant([np.random.randn(), np.random.randn(), 0.1*np.random.randn()]) #old
	random_translation = tf.constant([np.random.rand()-0.5, np.random.rand()-0.5, 0.1*np.random.rand()-0.05]) #new
	c2 += random_translation

	#rotate scans
	# rot = R_tf(tf.constant([0., 0., 0.05]))
	random_rotation = 0.03*tf.constant([np.random.randn(), np.random.randn(), np.random.randn()])
	rot = R_tf(random_rotation)
	c2 = c2 @ rot.numpy() 

	try:
		it = ICET(cloud1 = c1, cloud2 = c2, fid = 110, niter = 6, 
			draw = False, group = 2, RM = False, DNN_filter = False)
	except:
		print("oops")
		fail_count += 1

	# it = ICET(cloud1 = c1, cloud2 = c2, fid = 100, niter = 20, 
	# 	draw = False, group = 2, RM = False, DNN_filter = False, x0 = it.X)

	ICET_estimates[i] = it.X
	
	#test 
	ICET_estimates[i,:3] += random_translation
	ICET_estimates[i,3:] += random_rotation
	
	ICET_pred_stds[i] = it.pred_stds

# np.save("MC_results/scene1_ICET_estimates_v8", ICET_estimates)
# np.save("MC_results/scene1_ICET_pred_stds_v8", ICET_pred_stds)
np.save("MC_results/scene1_ICET_estimates_NM_v5", ICET_estimates)
np.save("MC_results/scene1_ICET_pred_stds_NM_v5", ICET_pred_stds)

print("fail count", fail_count)