from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET


## load custom point cloud geneated in matlab------------------------------------------
cloud1 = np.loadtxt("scene1_scan1.txt", dtype = float)
cloud2 = np.loadtxt("scene1_scan2.txt", dtype = float)
cloud1 = cloud1[cloud1[:,2] > -1.55] #ignore ground plane
cloud2 = cloud2[cloud2[:,2] > -1.55] #ignore ground plane
## ------------------------------------------------------------------------------------


niter = 100
ground_truth = np.array([-0.5, 0, 0, 0, 0, -0.035])

estimates = np.zeros([niter, 6])
pred_stds = np.zeros([niter, 6])

for i in range(niter):
	print("\n epoch", i ,"~~~~~~~~~~~~~~~~~~ \n")

	#add noise (if not generated when point clouds were created)
	c1 = cloud1 + 0.01*np.random.randn(np.shape(cloud1)[0], 3)
	c2 = cloud2 + 0.01*np.random.randn(np.shape(cloud2)[0], 3)

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 30, niter = 10, draw = False)

	estimates[i,:] = it.X.numpy()
	pred_stds[i,:] = it.pred_stds.numpy()

mean_soln = np.mean(estimates, axis = 0)
mean_error = np.mean(estimates - ground_truth, axis = 0)
soln_std = np.std(estimates, axis = 0)
mean_pred_std = np.mean(pred_stds, axis = 0)

print("\n ~~~~~~~~~~~~~~~~~~ \n")
print("\n correct soln", ground_truth)
print("\n mean_soln", mean_soln)
print("\n mean_error", mean_error)
print("\n soln_std", soln_std)
print("\n mean_pred_std", mean_pred_std)

