import numpy as np
from vedo import *
import vtk
import os
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import time
from utils import *

def ICET3D(Q, P, plt, bounds, fid, draw = False, num_cycles = 1):

	"""3D implementation of ICET algorithm using TensorFlow library"""

	#subdivide keyframe scan
	E1 = subdivide_scan_tf(Q, plt, bounds, fid, draw = False)
	mu1 = E1[0]
	sigma1 = E1[1]
	npts1 = E1[2]

	U, L = get_U_and_L(sigma1, bounds, fid)


	return(U, L)


def get_U_and_L(sigma1, bounds, fid):
	"""U = rotation matrix for each voxel to transform scan 2 distribution
				 into frame of corresponding to ellipsoid axis in keyframe
	   L = matrix to prune extended directions in each voxel (from keyframe)"""


	#use bounds and fid to calculate cellsize
	cellsize = tf.Variable([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])/tf.cast(fid, tf.float32)
	# print("\n cellsize \n", cellsize)

	eigenval, eigenvec = tf.linalg.eig(sigma1)
	U = tf.math.real(eigenvec)
	print("\n U \n",U)

	# simple length of each of the ellipses' 3 axis
	axislen = tf.math.real(eigenval) 
	print("\n axislen \n", axislen)

	# get projections of axis length in each direction
	print("\n tf.math.abs(tf.linalg.matmul(axislen, U)) \n",tf.math.abs(tf.linalg.matmul(axislen, U)))

	print(tf.math.greater(tf.math.sqrt(tf.abs(tf.linalg.matmul(axislen, U))), cellsize)) #generating a [4,4,3], needs to be [4,3,3]

	#generate L matrix by checking for overly extended axis directions

	L = None
	return(U, L)