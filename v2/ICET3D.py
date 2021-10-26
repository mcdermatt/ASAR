import numpy as np
from vedo import *
import vtk
import os
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import time
from utils import *

def ICET3D(pp1, pp2, plt, bounds, fid, draw = False, num_cycles = 5, min_num_pts = 30):

	"""3D implementation of ICET algorithm using TensorFlow library
	
			pp1 = point positions in first scan [M,3]
			pp2 = point positions in second scan [N,3]

	"""

	#subdivide keyframe scan
	E1 = subdivide_scan_tf(pp1, plt, bounds, fid, draw = False)
	mu1 = E1[0]
	sigma1 = E1[1]
	npts1 = E1[2]

	# ignore data from unused voxels
	nonzero_idx1 = tf.where(tf.math.reduce_sum(mu1, axis = 1) != 0)
	y0 = tf.squeeze(tf.gather(mu1,nonzero_idx1))
	sigma1 = tf.squeeze(tf.gather(sigma1, nonzero_idx1))

	# ignore voxels with too few points (this needs to be separate step)
	enough_pts1 = tf.where(npts1 > min_num_pts)
	npts1 = tf.squeeze(tf.gather(npts1,enough_pts1))
	y0 = tf.squeeze(tf.gather(y0, enough_pts1))
	sigma1 = tf.squeeze(tf.gather(sigma1, enough_pts1))

	# calculte overly extended directions for each remaining distribution  
	U, L = get_U_and_L(sigma1, bounds, fid)

	#TODO: visualize U and L
	#		replace ellipsoids with arrows in compact axis directions??

	#init solution vector x
	x = tf.zeros(3)

	#start with pp2_corrected with an initial transformation of [0,0,0]
	pp2_corrected = pp2[:]

	for i in range(num_cycles):

		#subdivide second scan
		E2 = subdivide_scan_tf(pp2_corrected, plt, bounds, fid, draw=False)
		mu2 = E2[0]
		sigma2 = E2[1]
		npts2 = E2[2]

		#remove all unused voxels
		nonzero_idx2 = tf.where(tf.math.reduce_sum(mu2, axis = 1) != 0)
		y = tf.squeeze(tf.gather(mu2,nonzero_idx2)) #with zero values removed
		sigma2 = tf.squeeze(tf.gather(sigma2, nonzero_idx2))

		# ignore voxels with too few points
		enough_pts2 = tf.where(npts2 > min_num_pts)
		npts2 = tf.squeeze(tf.gather(npts2,enough_pts2))
		y = tf.squeeze(tf.gather(y, enough_pts2))
		sigma2 = tf.squeeze(tf.gather(sigma2, enough_pts2))

		#determine correspondences between distribution centers of the two scans
		# print("\n shapes of y0 and y \n", tf.shape(y0), tf.shape(y))
		corr = get_correspondences_tf(y, y0)
		# print(corr)

		#reorder y0, U, and L according to correspondneces


		#Reshape y, y0 to be [3*N, 1] <- TODO: see if I actually need to do this...

		print(i)

	return(U, L)


def get_U_and_L(sigma1, bounds, fid):
	"""U = rotation matrix for each voxel to transform scan 2 distribution
				 into frame of corresponding to ellipsoid axis in keyframe
	   L = matrix to prune extended directions in each voxel (from keyframe)"""

	#TODO- calculate threshold for truncating axis lengths

	#use bounds and fid to calculate cellsize
	cellsize = tf.Variable([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])/tf.cast(fid, tf.float32)
	# print("\n cellsize \n", cellsize)

	eigenval, eigenvec = tf.linalg.eig(sigma1)
	U = tf.math.real(eigenvec)
	# print("\n U \n",U)
	# print("\n eigenval \n", tf.math.real(eigenval))


	#need to create [N,3,3] diagonal matrices for axislens
	zeros = tf.zeros([tf.shape(tf.math.real(eigenval))[0]])
	# print("\n zeros \n", zeros)
	axislen = tf.Variable([tf.math.real(eigenval)[:,0], zeros, zeros,
						   zeros, tf.math.real(eigenval)[:,1], zeros,
						   zeros, zeros, tf.math.real(eigenval)[:,2]])

	axislen = tf.reshape(tf.transpose(axislen), (tf.shape(axislen)[1], 3, 3))
	# print("\n axislen \n", axislen)

	# get projections of axis length in each direction
	# rotated = U @ axislen <- look up benefits of @ operator
	rotated = tf.abs(tf.matmul(U,axislen))
	# print("\n rotated \n", tf.squeeze(rotated))

	#check for overly extended axis directions
	thresh = cellsize #temp
	greater_than_thresh = tf.math.greater(rotated, thresh)
	# print("\n rotated > ___", greater_than_thresh)

	# geneate L as a ragged tensor --------------------------------------------------------------
	#get indices where greather_than_thresh == True
	ext_idx = tf.math.reduce_any(greater_than_thresh, axis = 1) #TODO -> make sure I am reducing about correct axis
	# print("\n ext_idx \n", ext_idx) 
	ext_idx = tf.where(tf.math.reduce_any(tf.reshape(ext_idx, (-1,1)), axis = 1) == False)
	# print("\n ext_idx \n", ext_idx)

	#create [3*N,3] identiy matrix
	L = tf.tile(tf.eye(3), (tf.shape(U)[0], 1))

	#only keep non-extended indices
	L = tf.squeeze(tf.gather(L, ext_idx))
	# print("\n L \n", L)

	#turn to ragged tensor with from_row_splits(?)
	# first (smallest) eigenvalue is (almost) never going to be overly extended 
	#		therefore, a row of [1,0,0] always signifies the start of a new voxel
	# print(tf.cast((tf.where(L[:,0] == 1)[:,0]), tf.int32))
	limits = tf.squeeze(tf.concat((tf.cast((tf.where(L[:,0] == 1)[:,0]), tf.int32), [tf.shape(L)[0]]), axis = 0))
	# print("\n Limits \n",limits)
	L = tf.RaggedTensor.from_row_limits(L,limits)[1:] #double counds first voxel without [1:]
	# print("\n L \n", L)
	# print("\n L \n", L.to_tensor())

	#----------------------------------------------------------------------------------------

	#	TODO: debug- make sure I should be getting rid of rows, not columns

	return(U, L)