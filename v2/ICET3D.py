import numpy as np
from vedo import *
import vtk
import os
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import time
from utils import *

#Optimization:
#TODO: 	figure out why memory usage is increasing after each loop
			# https://stackoverflow.com/questions/44825360/tensorflows-memory-cost-gradually-increasing-in-very-simple-for-loop/44825824

#Viz:
#TODO: 	remove past iterations of point cloud in viz
#TODO: 	add slider to allow selection of iterations
#			generate every <x> beforehand 
#TODO:	visualize L1

#Algorithm: 
#TODO: 	figure out why I need to make angs[0] and angs[1] negative
#TODO:	Make sure voxels with insufficeint number of points are getting ignored
#TODO:	Implement cutoff threshold values for L1


def ICET3D(pp1, pp2, plt, bounds, fid, test_dataset = False,  draw = False, 
	       num_cycles = 5, min_num_pts = 50, draw_grid = False, draw_ell = True, 
	       draw_corr = False, CM = "voxel"):

	"""3D implementation of ICET algorithm using TensorFlow library
	
			pp1 = point positions in first scan [M,3]
			pp2 = point positions in second scan [N,3]

	"""

	if test_dataset == True:
		pp1, pp2, bounds = generate_test_dataset()

	#subdivide keyframe scan
	if draw == True:
		E1 = subdivide_scan_tf(pp1, plt, bounds, fid, draw = True, show_pc = 1, draw_grid = draw_grid, draw_ell = draw_ell)
	else:
		E1 = subdivide_scan_tf(pp1, plt, bounds, fid, draw = False, draw_grid = draw_grid)
	mu1 = E1[0]
	sigma1 = E1[1]
	npts1 = E1[2]
	disp1 = E1[3]

	#TODO: DEBUG -> I think there is a mismatch between npts1 and its corresponding elements in sigma1, etc.
	# print("\n npts1 \n", npts1)

	# ignore data from unused voxels
	nonzero_idx1 = tf.where(tf.math.reduce_sum(mu1, axis = 1) != 0)
	y0 = tf.squeeze(tf.gather(mu1,nonzero_idx1))
	sigma1 = tf.squeeze(tf.gather(sigma1, nonzero_idx1))

	# ignore voxels with too few points (this needs to be separate step)
	enough_pts1 = tf.where(npts1 > min_num_pts)
	npts1 = tf.squeeze(tf.gather(npts1,enough_pts1))
	y0 = tf.squeeze(tf.gather(y0, enough_pts1))
	sigma1 = tf.squeeze(tf.gather(sigma1, enough_pts1))
	# print("\n enough_pts1 \n", npts1)

	# calculte overly extended directions for each remaining distribution  
	U, L = get_U_and_L(sigma1, bounds, fid)

	#init solution vector x
	x = tf.zeros(6) #[x, y, z, phi, theta, psi].T

	#start with pp2_corrected with an initial transformation of [0,0,0]
	# pp2_corrected = pp2[:] #was this
	t = x[:3]
	rot = R_tf(x[3:])
	pp2_corrected = tf.matmul(pp2, rot) + t

	x_hist = tf.zeros([1,6])

	for i in range(num_cycles):
		print(i)

		#subdivide second scan
		if draw == True:
			#referencing disp 1 here prevents re-drawing old transformations of scan2
			E2 = subdivide_scan_tf(pp2_corrected, plt, bounds, fid, disp = disp1, draw=True, show_pc = 2, draw_grid = draw_grid, draw_ell = draw_ell)
		else:
			E2 = subdivide_scan_tf(pp2_corrected, plt, bounds, fid, draw=False)
		mu2 = E2[0]
		sigma2 = E2[1]
		npts2 = E2[2]
		disp = E2[3]

		#remove all unused voxels
		nonzero_idx2 = tf.where(tf.math.reduce_sum(mu2, axis = 1) != 0)
		y = tf.squeeze(tf.gather(mu2,nonzero_idx2)) #with zero values removed
		sigma2 = tf.squeeze(tf.gather(sigma2, nonzero_idx2))

		# ignore voxels with too few points
		enough_pts2 = tf.where(npts2 > min_num_pts)
		# print("\n enough_pts2 \n", tf.gather(npts2, enough_pts2))
		npts2 = tf.squeeze(tf.gather(npts2,enough_pts2))
		y = tf.squeeze(tf.gather(y, enough_pts2))
		sigma2 = tf.squeeze(tf.gather(sigma2, enough_pts2))

		#determine correspondences between distribution centers of the two scans
		# print("\n shapes of y and y0 \n", tf.shape(y), tf.shape(y0)) #TODO: debug here
		if draw_corr == True:
			corr, disp = get_correspondences_tf(y, y0, bounds, fid, method = CM, disp = disp, draw_corr = True)
		else:
			corr = get_correspondences_tf(y, y0, bounds, fid, method = CM)
		# print(corr)

		#ignore all data from voxels in y where corresponance does not exist (does nothing if method = "NN")
		y_i = tf.gather(y, corr[:,1])
		sigma2_i = tf.gather(sigma2, corr[:,1])
		npts2_i = tf.gather(npts2, corr[:,1])

		#reorder y0, U, L, npts1, and cov1 according to correspondneces
		y0_i = tf.gather(y0, corr[:,0])
		U_i = tf.gather(U, corr[:,0])
		L_i = tf.gather(L, corr[:,0])
		npts1_i = tf.gather(npts1, corr[:,0])#[:,None]
		sigma1_i = tf.gather(sigma1, corr[:,0])

		#Reshape y, y0 to be [3*N, 1] <- TODO: see if I actually need to reshape
		# print("\n y0_i shape \n", tf.shape(y0_i))
		# print("\n y shape \n", tf.shape(y))

		#get matrix containing partial derivatives for each voxel mean
		H = jacobian_tf(tf.transpose(y_i), x[3:]) # shape = [num of corr * 3, 6]
		# print("before reshape", H)
		H = tf.reshape(H, (tf.shape(H)[0]//3,3,6)) # -> need shape [#corr//3, 3, 6]
		# print("after reshape", H)
		#H is correct


		#get dx--------------------------------------------------------------------
		R_noise = (tf.transpose(tf.transpose(sigma1_i) / tf.cast(npts1_i - 1, tf.float32)) + 
				   tf.transpose(tf.transpose(sigma2_i) / tf.cast(npts2_i - 1, tf.float32)) )
		# R_noise = L_i * U_i.T * R_noise * U_i * L_i.T
		R_noise = L_i.to_tensor() @ tf.transpose(U_i, [0,2,1]) @ R_noise @ U_i @ tf.transpose(L_i.to_tensor(), [0,2,1])
		# print("\n R_noise \n", R_noise)

		#transpose each [i,3,3] element of U_i here
		U_iT = tf.transpose(U_i, [0,2,1])
		# print("\n U_iT \n", tf.shape(U_iT)) 		  #[19, 3, 3]
		# print("\n L_i \n", tf.shape(L_i.to_tensor())) #[19, 2, 3] with only [5,5,2] fidelity
		LUT = tf.math.multiply(L_i.to_tensor(), U_iT) #TODO -> FIX BUG HERE
		#NOTE: this bug happens when the every voxel has at least one ambigious direction
		# print("\n LUT \n", tf.shape(LUT))

		# print("\n LUT \n", tf.shape(LUT))
		H_z = tf.matmul(LUT,H)
		# H_z = tf.tensordot(LUT,H, axes = (1,2))
		# print(tf.shape(H_z))

		#invert sensor noise matrix R to get weighting matrix W
		W = tf.linalg.pinv(R_noise)
		# print("\n W \n",W)

		HTWH = tf.math.reduce_sum(tf.matmul(tf.matmul(tf.transpose(H_z, [0,2,1]), W), H_z), axis = 0)
		# print("\n HTWH \n", HTWH)

		HTW = tf.matmul(tf.transpose(H_z, [0,2,1]), W)
		# print("\n HTW \n",tf.shape(HTW))

		#check condition number
		L2, lam, U2 = check_condition(HTWH)
		# print("\n L2 \n", L2)

		# create alternate corrdinate system to align with axis of scan 1 distributions
		z = tf.squeeze(tf.matmul(LUT, y_i[:,:,None]))
		z0 = tf.squeeze(tf.matmul(LUT, y0_i[:,:,None]))	
		dz = z - z0
		dz = dz[:,:,None] #need to add an extra dimension to dz to get the math to work out
		# print("\n dz \n", dz) #looks fine, most differences are between 0.1-1 units

		#TODO -> HTW is WAAAY too big (?)

		# # #solve for dx - with L2 pruning ----------------------------------------------------------
		# # dx     = (L2 * U2.T)^-1       * L2    * U2     * HTW         *  dz
		# # [6, 1] = ([D, 6] * [6, 6])^-1 * [D,6] * [6, 6] * [B, 6, 3] * [B,3]
		# #	   D = 1-6 depending on # axis removed 
		# #	   B = batch size (num usable voxels)

		# #not sure where this came from but it's wrong...
		# # dx = tf.squeeze(tf.matmul(tf.matmul(tf.linalg.pinv(L2 @ tf.transpose(U2)) @ L2 @ tf.transpose(U2), HTW), dz))

		# #trying this
		# dx = tf.squeeze(tf.matmul(tf.matmul( (tf.linalg.pinv(L2 @ tf.transpose(U2)) @ L2 @ tf.transpose(U2) , HTW ) )), dz) #rank deficient

		# #need to add up the tensor containing the summands from each voxel to a single row matrix
		# #    [B, 6] -> [6]
		# dx = tf.math.reduce_sum(dx, axis = 0)
		# print("\n dx \n", dx)
		# # #-----------------------------------------------------------------------------------------

		#test - solve for dx without L2 pruning --------------------------------------------------
		#dx = (HTWH)^-1 * [HTW] * dy
		dx = tf.linalg.pinv(HTWH) @ HTW @ (y_i - y0_i)[:,:,None]
		dx = tf.squeeze(tf.math.reduce_sum(dx, axis = 0))
		# print("\n dx \n", dx)
		#-----------------------------------------------------------------------------------------

		#get output covariance matrix
		Q = tf.linalg.pinv(HTWH)
		# print("\n Q \n", Q)

		#augment x by dx
		x -= dx
		# x = x + dx
		print("\n x \n", x)

		#transform 2nd scan by x
		# t = x[:3]
		# rot = R_tf(x[3:])

		# # DEBUG: only consider yaw transformations
		t = tf.constant([0., 0., 0.])
		rot = R_tf(tf.constant([0., 0., x[5].numpy()]))	

		# print("\n t, rot \n", t, "\n", rot)

		#update pp2
		pp2_corrected = tf.matmul(pp2, rot) + t #TODO: fix this step
		# pp2_corrected = tf.matmul((pp2 + t), rot)

		# print(tf.shape(pp2_corrected))

		#update solution history
		x_hist = tf.concat((x_hist, x[None,:]), axis = 0)
		# print("x_hist" ,x_hist)

		#--------------------------------------------------------------------------

		if draw == True:
			# plt.clear(at = 0) #removes axis
			# if i > 0:
				# plt.pop(at=0)
				# plt.clear(disp, at=0) 
			plt.show(disp, "ICET3D", at=0, interactive = True) 
			#set interactive to False to autoplay
			#NOTE: vedo documentation is incorrect, plotter does NOT have <new> parameter
			# new = True opens new scans in seperate window


	# print("\n x \n", x)
	return(Q, x_hist)


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
	rotated = tf.abs(tf.matmul(U,axislen))
	# print("\n rotated \n", tf.squeeze(rotated))

	#check for overly extended axis directions
	thresh = cellsize/2 #temp
	# thresh = cellsize*2 #will not truncate anything?
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

	#	TODO: debug- make sure L should be getting rid of rows, not columns

	return(U, L)


def check_condition(HTWH):
	"""verifies that HTWH is invertable and if not, 
		reduces dimensions to make inversion possible

		L2 = identity matrix which keeps non-extended axis of solution
		lam = diagonal eigenvalue matrix
		U2 = rotation matrix to transform for L2 pruning 
		"""

	#TODO: keep L2 as a 6x6- Yes or no?

	cutoff = 10e9 #10e5

	#do eigendecomposition
	eigenval, eigenvec = tf.linalg.eig(HTWH)
	eigenval = tf.math.real(eigenval)
	eigenvec = tf.math.real(eigenvec)

	# print("\n eigenvals \n", eigenval)
	# print("\n eigenvec \n", eigenvec)


	#sort eigenvals by size -default sorts small to big
	# small2big = tf.sort(eigenval)
	# print("\n sorted \n", small2big)

	#test if condition number is bigger than cutoff
	condition = eigenval[-1] / eigenval[0]

	everyaxis = tf.cast(tf.linspace(0,5,6), dtype=tf.int32)
	remainingaxis = everyaxis
	i = tf.Variable([0],dtype = tf.int32) #count var
	#loop until condition number is small enough to make matrix invertable
	while condition > cutoff:

		condition = eigenval[-1] / tf.gather(eigenval, i)

		i.assign_add(tf.Variable([1],dtype = tf.int32))

		remainingaxis = everyaxis[i.numpy()[0]:]

		# print(remainingaxis)

	#create identity matrix truncated to only have the remaining axis
	L2 = tf.gather(tf.eye(6), remainingaxis)
	# print("\n L2 \n", L2)

	U2 = eigenvec
	# print("\n U2 \n", U2)


	lam = tf.eye(6)*eigenval
	# print("\n lam \n", lam)

	return(L2, lam, U2)