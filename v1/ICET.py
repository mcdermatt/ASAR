import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import misc
from utils import *
from NDT import NDT
from ICP import ICP_least_squares

#TODO
#	Verify input dimensions are correct!!!

def remove_ambiguity(Q):

	"""removes axis in which ellipses stretch outside voxels """

	L = np.array([[0, 1]])

	z = None1

	return Q

def get_H(P, x):

	"""returns array H containing jacobians for each voxel
		
		P: Input Points
		x: [x, y, theta].T
	"""

	#was this...
	H = np.zeros([np.shape(P)[1]*2,3]) 
	for i in range(np.shape(P)[1]):
		H[2*i:2*i+2] = jacobian(x,P[:,i])

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

	#extract center data from E1, E2
	ctr1 = np.zeros([len(E1),2])
	for idx1, c1 in enumerate(E1):
		ctr1[idx1,:] = c1[0]
	# print("ctr1: \n", ctr1)
	ctr2 = np.zeros([len(E2),2])
	for idx2, c2 in enumerate(E2):
		ctr2[idx2,:] = c2[0]


	#get correspondences
	correspondences = get_correspondence(ctr2.T, ctr1.T, fig, ax, draw = True)
	# print("correspondences: \n", correspondences)

	y = ctr2 #just use new measuremnet as y
	y0 = ctr1[correspondences[0].astype(int)]
	# print("ctr2 = y: \n", y)
	# print("y0: \n", y0)

	#reshape Ys to be [ _ , 1]
	y_reshape = np.reshape(y, (np.shape(y)[0]*2,1), order='F')
	y0_reshape = np.reshape(y0, (np.shape(y0)[0]*2,1), order='F')

	#inital estimate for transformation
	x = np.zeros([3,1])

	for cycle in range(num_cycles):

		# print("y : \n", y, np.shape(y))

		H = get_H(y.T, x)
		# print("H: \n", H, np.shape(H))

		#init W (set as identity matrix for now)
		W = np.identity(np.shape(H)[0])

		H_w = weighted_psudoinverse(H, W)
		# print("H_w: \n", np.shape(H_w))


		dx = H_w.dot(y_reshape - y0_reshape)
		print("dx = ", dx)

		# y_reshape = y0_reshape + H.dot(dx)

		x -= dx


	#draw first 2nd point cloud with transformation applied
	rot = R(x[2])
	t = x[0:2]
	P_corrected = rot.dot(pp2.T) + t
	ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.0625), ls = '', marker = '.', markersize = 20)

	return x

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