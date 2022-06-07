import numpy as np
import tensorflow as tf
#need to have these two lines to work on my ancient 1060 3gb
#  https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from utils import *
import tensorflow_probability as tfp
import time
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
from ICET3D import ICET3D

""" Script to generate transformation estimates FROM ENTIRE TRAJECTORIES
 (Spherical ICET Paper) using old cartesian method  as baseline"""

scene = 1   #scene 1 = highway, scene 2 = mountain
nframes = 40 #40 #how many sequential frames to consider (num pairs = num frames - 1)
niter = 3   #number of times to repeat each pair of scans


#init params of cartesian ICET
nc = 10	 #number of cycles
mnp = 10 #50#100 #minimum number of points per voxel
D = False #draw sim
DG = True #draw grid
DE = False #draw ellipsoids
DC = True #draw correspondences
TD = False #use test dataset
CM = "voxel" #correspondence method, "voxel" or "NN"
vizL = True #draw arrows in direction of non-truncated directions for each distribution
plt = Plotter(N=1, axes=4, interactive=True)

#init struct to store results of each sim
ICET_estimates = np.zeros([niter*(nframes - 1), 6])
ICET_pred_stds = np.zeros([niter*(nframes - 1), 6])


for idx in range(1, nframes):
	for i in range(niter):
		print("------------- Epoch ", (idx-1)*niter + i, "---------------")

		fn1 = "C:/Users/Derm/vaRLnt/v3/MC_trajectories/scene" + str(scene) + "_scan" + str(idx) + ".txt" 
		fn2 = "C:/Users/Derm/vaRLnt/v3/MC_trajectories/scene" + str(scene) + "_scan" + str(idx + 1) + ".txt" 
		c1_raw = np.loadtxt(fn1, dtype = float)
		c2_raw = np.loadtxt(fn2, dtype = float)

		#add noise (if not generated when point clouds were created)
		c1 = c1_raw + 0.02*np.random.randn(np.shape(c1_raw)[0], 3)
		c2 = c2_raw + 0.02*np.random.randn(np.shape(c2_raw)[0], 3) 

		if scene == 1:
			lim = tf.constant([-60., 60., -20., 20., -5.,5.])
			f = tf.constant([18,6,6])

			#Problem seems to be points that are on outer rings of scan
			#  -> need to remove any points in scan1 > 25m(?) from center 
			c1 = c1[c1[:,0] > -30]
			c1 = c1[c1[:,0] < 30]
			c1 = c1[c1[:,2] > -1.25] #ignore ground plane
			c2 = c2[c2[:,2] > -1.25] #ignore ground plane

		if scene == 2:
			lim = tf.constant([-100., 100., -100., 100., -30.,30.])
			f = tf.constant([40,40,1])

		rot = R_tf(tf.constant([0., 0., 0.05]))
		c2 = c2 @ rot.numpy() 

		cloud1_tensor = tf.convert_to_tensor(c1, dtype = tf.float32)
		cloud2_tensor = tf.convert_to_tensor(c2, dtype = tf.float32)


		Q, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = lim, 
				fid = f, num_cycles = nc , min_num_pts = mnp, draw = D, draw_grid = DG,
				draw_ell = DE, draw_corr = DC, test_dataset = TD, CM = CM, vizL = vizL)

		ICET_estimates[(idx-1)*niter + i] = x_hist[-1] 
		ICET_pred_stds[(idx-1)*niter + i] = np.sqrt(np.abs(np.diag(Q)))

# np.save(r"C:\Users\Derm\vaRLnt\v3\MC_results\traj1_cart_ICET_estimates_with_GP", ICET_estimates)
# np.save(r"C:\Users\Derm\vaRLnt\v3\MC_results\traj1_cart_ICET_pred_stds_with_GP", ICET_pred_stds)

np.save(r"C:\Users\Derm\vaRLnt\v3\MC_results\traj1_cart_ICET_estimates_no_GP", ICET_estimates)
np.save(r"C:\Users\Derm\vaRLnt\v3\MC_results\traj1_cart_ICET_pred_stds_no_GP", ICET_pred_stds)

# np.save(r"C:\Users\Derm\vaRLnt\v3\MC_results\traj2_cart_ICET_estimates", ICET_estimates)
# np.save(r"C:\Users\Derm\vaRLnt\v3\MC_results\traj2_cart_ICET_pred_stds", ICET_pred_stds)
