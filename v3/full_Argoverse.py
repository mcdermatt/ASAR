from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
import pyarrow.feather as feather

num_frames = 155
NDT_estimates = np.zeros([num_frames, 6])
# OXTS_baseline = np.zeros([num_frames, 6])
# ICET_pred_stds = np.zeros([num_frames, 6])


#get ground truth to seed initial translation estimates
gt_lidar = np.loadtxt("spherical_paper/Argoverse_results/gt_lidar.txt")
dgt_lidar = np.diff(gt_lidar[:,:2], axis = 0)
true_fwd = np.sqrt(np.sum(dgt_lidar**2, axis = 1)) #absolute movement per frame in horizontal plane


path = "D:/sensor/train/087fec73-1a0c-399a-9292-cc2cf99dc97f/sensors/lidar/" #urabn canyon
files = os.listdir(path)

fn1 = path + files[0]

count = 0
for i in files[1:]:
	fn2 = path + i

	print("---------------------", count, "---------------------")
	#convert feather file to dataframe object
	df1 = feather.read_feather(fn1)
	df2 = feather.read_feather(fn2)

	pts1 = df1[['x', 'y', 'z']].to_numpy()
	pts2 = df2[['x', 'y', 'z']].to_numpy()

	#save file 1 to text (for matlab benchmarking)
	# np.savetxt("D:/argoverse_benchmarks/urban_canyon/scan" + str(count) + ".txt", pts1)

	#remove ground plane
	ground = 0.3 
	pts1 = pts1[pts1[:,2] > ground ]
	pts2 = pts2[pts2[:,2] > ground ]

	initial_guess = tf.cast(tf.constant([true_fwd[count] + 0.1*np.random.randn() , 0., 0., 0., 0., 0.]), tf.float32)
	it = ICET(cloud1 = pts1, cloud2 = pts2, fid = 70, niter = 12, draw = False, group = 2, 
		RM = True, DNN_filter = False, x0 = initial_guess)
	NDT_estimates[count,:] = it.X

	fn1 = fn2
	count += 1

#save last file to text (for matlab benchmarking)
# np.savetxt("D:/argoverse_benchmarks/urban_canyon/scan" + str(count + 1) + ".txt", pts2)

np.savetxt("Argoverse_NDT_spherical_nogp.txt", NDT_estimates)