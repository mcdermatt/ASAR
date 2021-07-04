import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import misc
from utils import *
from NDT import NDT
from ICP import ICP_least_squares

#TODO
#	debug correspondences in ICET_v2
#		it looks like I'm getting coors between 1st scan and translated 1st scan...
#	turn iterative x values into one final transformation

#Ideas
#	Use both NN AND zero for initial estimate of x
#		if NN worse than zero, ignore it


def get_weighting_matrix(cov):

	'''
	cov: 3D matrix containing covariance matrices for all voxels 

	R_noise: sensor noise model, (should reflect the spread of points due to ______)
		R = Q /(|J|), Q = true covariance of voxel J, |J| = # pts in J
			size(R) = [J,J] where J is the total number of voxels

	'''
	R_noise = np.identity(np.shape(cov)[0]*2) #multiply by 2 because cov matrices are 2x2

	#TODO - need to take in information on the number of point inside each ellipse
	pts_in_cell = 1

	for i in range(np.shape(cov)[0]):
		R_noise[(2*i):(2*i+2),(2*i):(2*i+2)] = cov[i] / (pts_in_cell) #-1

	# print(np.floor(R_noise[:12,:12])) #make sure everything looks like the right shape

	W = np.linalg.inv(R_noise)

	return W


def get_H(P, x):

	"""returns array H containing jacobians for each voxel
		
		P: Input Points
		x: [x, y, theta].T
	"""

	#works when P is [ n , 2] array ------------------------
	# H = np.zeros([np.shape(P)[1]*2,3]) 
	# for i in range(np.shape(P)[1]):
	# 	H[2*i:2*i+2] = jacobian(x,P[:,i])

	#if P is a [2n , 1] array ------------------------------
	H = np.zeros([np.shape(P)[0],3])
	# print("P = ", P , np.shape(P))
	for i in range(np.shape(P)[0]//2):
		H[2*i:2*i+2] = jacobian(x, P[2*i:2*i+2,0])
		# print(H)

	return H

def weighted_psudoinverse(H, W = np.identity(3)):

	""" Useful for "solving" an overdetermined system that has more equations than unknowns

	Ex: system of 3 equations and 2 unknowns which has imperfect measurements
		
		We want to minimize || Ax - b ||^2 -> to get as close to Ax = b as possible
			solutions are given by: (A^T)Ax = (A^T)b

		if columns of A are linearly independant: (A^T)A is invertable
			therefore: x = inv( (A^T)A )(A^T)b 	

	H = Array containing appended Jacobians for each voxel 
		  = [ H(1), 
		  	  H(2),
		  	   ..., 
		  	  H(J) ]
	W = weighting matrix
	  = R^-1, where R is the sensor noise covariance matrix

	  		R = Q / (|J|)
	  			 Q -> True covariance of voxel j
	  			|J| -> number of points in voxel j

	https://sci.utah.edu/~gerig/CS6640-F2012/Materials/pseudoinverse-cis61009sl10.pdf
	"""

	#H_w = H^w
	H_w = np.linalg.inv(H.T.dot(W).dot(H)).dot(H.T).dot(W)

	return H_w

def ICET_v2(Q,P,fig,ax,fid = 10, num_cycles = 1, draw = True):

	"""similar to v1 except uses "weighted" psudoinverse instead of LSICP"""

	#get point positions in 2d space and draw 1st and 2nd scans
	pp1 = draw_scan(Q,fig,ax, pt = 0)
	pp2 = draw_scan(P,fig,ax, pt = 1) #pt number assigns color for plotting
	#group points into ellipses
	E1 = subdivide_scan(pp1,fig,ax, fidelity = fid, pt = 0)
	E2 = subdivide_scan(pp2,fig,ax, fidelity = fid, pt = 1)

	#extract: center data from E1, E2 -> center points are a 2d array
	#         covariance data from E1, E2 
	ctr1 = np.zeros([len(E1),2])
	cov1 = np.zeros([len(E1),2,2])
	for idx1, c1 in enumerate(E1):
		ctr1[idx1,:] = c1[0]
		cov1[idx1,:] = c1[1]
	ctr2 = np.zeros([len(E2),2])
	cov2 = np.zeros([len(E2),2,2])
	for idx2, c2 in enumerate(E2):
		ctr2[idx2,:] = c2[0]
		cov2[idx2,:] = c2[1]
	# print("cov2: \n", cov2, np.shape(cov2))

	#get weighting matrix from covariance matrix -----------------------------------------------------
	# 	#TODO: does this need to be iteratively updated?? -no I think?
	# W = get_weighting_matrix(cov2)
	W = np.identity(np.shape(ctr2)[0]*2) #debug: simple identity for W
	print("W: \n", W[:6,:6])

	#inital estimate for transformation
	x = np.zeros([3,1])
	#init total x of the system (DEBUG: get rid of this variable, x should encode all required information)
	total_transformation = np.zeros([3,1])

	y = ctr2
	y0 = ctr1
	P_corrected = pp2

	#TODO: improve initial estimated transform (start with zeros here, could potentially use wheel odometry) 
		  #call on nerual network to get inital estimate x


	# ax.plot(y.T[0,:], y.T[1,:], color = (1,1,1,1.), ls = '', marker = '.', markersize = 10)

	for cycle in range(num_cycles):

		#get correspondences needs to take in 2d array of points
		correspondences = get_correspondence(y.T, y0.T, fig, ax, draw = False)
		# print("correspondences: \n", correspondences)

		y0 = y0[correspondences[0].astype(int)]
		# print("y0: \n", y0, np.shape(y0))
		# print("y: \n", y, np.shape(y0))

		#reshape Ys to be [ _ , 1]
		y_reshape = np.reshape(y, (np.shape(y)[0]*2,1), order='C') #was F order -> wrong
		y0_reshape = np.reshape(y0, (np.shape(y0)[0]*2,1), order='C')
		# print("y0_reshape: \n", np.shape(y0_reshape))

		#trying this for y shape [2n, 1]
		H = get_H(y_reshape, x)
		# print("H = \n", H, np.shape(H))
		
		H_w = weighted_psudoinverse(H, W)
		# print("H_w: \n", np.shape(H_w))

		dx = H_w.dot(y_reshape - y0_reshape)
		print("error: \n", np.sum(abs(y_reshape- y0_reshape)))

		x -= dx
		print("dx = ", dx)
		total_transformation += x

		#incrementally update y
		rot = R(x[2])
		t = x[0:2]
		y = rot.dot(y.T) + t
		y = y.T

		#DEBUG: draw progression of transformation 
		ax.plot(y.T[0,:], y.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1.), ls = '', marker = '.', markersize = 10)
		print("x = \n", x)

		# #incrementally update P_corrected (for debug)
		# rot = R(x[2])
		# t = x[0:2]
		# P_corrected = rot.dot(P_corrected.T) + t
		# P_corrected = P_corrected.T

		# #draw all points progressing through transformation
		# ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),0.1), ls = '', marker = '.', markersize = 20)

	#draw final translated points (for debug)
	# ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1,0,0,0.0375), ls = '', marker = '.', markersize = 20)

	#draw final translated points using initial P and final X -> NOT WORKING
	rot = R(total_transformation[2])
	t = total_transformation[:2]
	P_final = rot.dot(pp2.T) + t
	ax.plot(P_final.T[:,0], P_final.T[:,1], color = (1,0,0,0.0125), ls = '', marker = '.', markersize = 20)

	return total_transformation

def ICET_v1(Q, P, fig, ax, fid = 10, num_cycles = 1, draw = True):

	"""
	Core algorithm:

	1) Apply an NDT from the reference point cloud to a mixture of Gaussian distributions
		with (at most) one normal distribution per voxel
	
	2) Repeat the NDT process for a later point cloud

	3) Use least squares to find the states (R, t) that match the distribution mean for each voxel
		as closely as possible
	"""

	#get point positions in 2d space and draw 1st and 2nd scans
	pp1 = draw_scan(Q,fig,ax, pt = 2)
	pp2 = draw_scan(P,fig,ax, pt = 2) #pt number assigns color for plotting

	E1 = subdivide_scan(pp1,fig,ax, fidelity = fid, pt = 0)
	E2 = subdivide_scan(pp2,fig,ax, fidelity = fid, pt = 1)

	#extract center data from E1, E2
	ctr1 = np.zeros([len(E1),2])
	for idx1, c1 in enumerate(E1):
		ctr1[idx1,:] = c1[0]

	ctr2 = np.zeros([len(E2),2])
	for idx2, c2 in enumerate(E2):
		ctr2[idx2,:] = c2[0]

	#TODO: Only consider subselection of voxels where valid mean is obtained for both clouds??


	P_corrected, t, rot = ICP_least_squares(ctr1.T, ctr2.T, fig, ax, num_cycles = num_cycles, draw = True)

	return t, rot

def get_state_error():

	P = None #TODO

	return P 

def remove_ambiguity(Q):

	"""removes axis in which ellipses stretch outside voxels """

	L = np.array([[0, 1]])

	z = None1

	return Q