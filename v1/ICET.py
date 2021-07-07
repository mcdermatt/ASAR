import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import misc
from utils import *
from NDT import NDT
from ICP import ICP_least_squares

#TODO:
#	figure out where to account for U and L

def get_U_and_L(cov1):

	"""generates matrices used to remove axis in which ellipses for each voxel are extended 
			overly extended axis is defined as:
			ax > (L^2)/16, where L is the length of the voxel 
		
		inptus: cov1 -> cov from ellipse data from first scan taken from subdivide_scan()[1]

		outputs: FOR EACH VOXEL 
				 U -> rotation matrix to align point with the major and minor axis of ellipse
				 L -> reduced dimension array used to ignore extended axis directions
	
			"""

	U = np.zeros([np.shape(cov1)[0],2,2])
	# L = np.zeros([np.shape(cov1)[0], 1]) #don't know what size L is going to be before we 
	L = [[]]

	#loop through every voxel in scan 1 (only do this once per keyframe)
	for i in range(np.shape(cov1)[0]):

		eig = np.linalg.eig(cov1[i])
		eigenval = eig[0]
		eigenvec = eig[1]

		#get new coordinate frame
		theta_temp = np.arcsin(eigenvec[0,1]/eigenvec[0,0])
		#get U matrix requred to rotate future P points so that x', y' axis align with major and minor axis of ellipse
		U[i] = R(theta_temp)

		# get L matrix 
		major = np.sqrt(eigenval[1])
		minor = np.sqrt(eigenval[0])

		#TODO: take in cellsize as a parameter from subdivide_scan()
		cellsize = 15
		#base case: no elongated directions
		if major < (cellsize/4)**2 and minor < (cellsize/4)**2:
			L_i = np.array([1,1])
		#elongated major
		if major > (cellsize/4)**2 and minor < (cellsize/4)**2:
			L_i = np.array([0,1])
		#elongated minor (should not ever happen)
		if minor > (cellsize/4)**2 and major < (cellsize/4)**2:
			L_i = np.array([1, 0])
		#elongated both
		if major > (cellsize/4)**2 and minor > (cellsize/4)**2:
			L_i = np.array([0,0])
	
		L = np.append(L, L_i)
		#TODO- not correct as is?? -> should be removing axis from L, not setting to zero

	# L = L[:,None]

	#BE CAREFUL:
	#	U and L are in relation to the origonal scan, but the coorespondences used in the main loop 
	#	will be in a different order


	return U, L

def get_weighting_matrix(cov, npts, U = 0, L = None):

	'''
	cov: 3D matrix containing covariance matrices for all voxels 
	
	npts: number of points inside each voxel of cov

	R_noise: sensor noise model, (should reflect the spread of points due to ______)
		R = Q /(|J|), Q = true covariance of voxel J, |J| = # pts in J
			size(R) = [J,J] where J is the total number of voxels

	'''
	R_noise = np.identity(np.shape(cov)[0]*2) #multiply by 2 because cov matrices are 2x2


	# print("U \n", np.shape(U))
	# print("L \n", np.shape(L))

	for i in range(np.shape(cov)[0]): 

		#normalize true covariance by the number of points in the subdivision
		M = cov[i] / (npts[i] - 1) 
		
		#account for U and L matrices - before inverse?
		if U.all() != None and L.all() != None:
			#NOTE: underscript _j denotes that this is the jth voxel in the scan
			L_j = L[2*i:(2*i+2)]
			U_j = U[i]

			#TODO: verify that this is the right place for accounting for U and L
			# M = L_j.dot(U_j.T.dot(M.dot(U_j.dot(L_j.T))))

		R_noise[(2*i):(2*i+2),(2*i):(2*i+2)] = M

	# print(np.floor(R_noise[:12,:12])) #make sure everything looks like the right shape

	# W = np.linalg.inv(R_noise) #does not work (singular matrix error)
	W = np.linalg.pinv(R_noise)

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
	# H_w = np.linalg.inv(H.T.dot(W).dot(H)).dot(H.T).dot(W)
	H_w = np.linalg.pinv(H.T.dot(W).dot(H)).dot(H.T).dot(W)

	return H_w

def ICET_v2(Q,P,fig,ax,fid = 10, num_cycles = 1, min_num_pts = 5, draw = True):

	"""similar to v1 except uses "weighted" psudoinverse instead of LSICP"""

	#get point positions in 2d space and draw 1st and 2nd scans
	pp1 = draw_scan(Q,fig,ax, pt = 0)
	pp2 = draw_scan(P,fig,ax, pt = 1) #pt number assigns color for plotting

	#group points into ellipses
	E1 = subdivide_scan(pp1,fig,ax, fidelity = fid, pt = 0, min_num_pts = min_num_pts)

	#extract: center data from E1, E2 -> center points are a 2d array
	#         covariance data from E1, E2 
	ctr1 = np.zeros([len(E1),2])
	cov1 = np.zeros([len(E1),2,2])
	npts1 = np.zeros(len(E1))
	for idx1, c1 in enumerate(E1):
		ctr1[idx1,:] = c1[0]
		cov1[idx1,:] = c1[1]
		npts1[idx1] = c1[2]

	#to avoid taking into account directions with extended axis
	U, L = get_U_and_L(cov1)

	#inital estimate for transformation
	#TODO: improve initial estimated transform (start with zeros here, could potentially use wheel odometry) 
	x = np.zeros([3,1])

	y0 = ctr1
	y0_init = ctr1 #hold on to initial centers so we don't lose information when doing correspondences
	P_corrected = pp2 #for debug

	for cycle in range(num_cycles):

		#Refit 2nd scan into voxels on each iteration
		E2 = subdivide_scan(P_corrected,fig,ax, fidelity = fid, pt = 2, min_num_pts = min_num_pts)

		ctr2 = np.zeros([len(E2),2])
		cov2 = np.zeros([len(E2),2,2])
		npts2 = np.zeros(len(E2))
		for idx2, c2 in enumerate(E2):
			ctr2[idx2,:] = c2[0]
			cov2[idx2,:] = c2[1]
			npts2[idx2] = c2[2]
		# print("cov2: \n", cov2, np.shape(cov2)) #cov2 may change size on every iteration as voxles recieve more or less than the required min_num_pts

		y = ctr2
		if cycle == 0:
			y_init = ctr2 #save for later translations

		#get correspondences needs to take in 2d array of points
		correspondences = get_correspondence(y.T, y0.T, fig, ax, draw = False)
		# print("correspondences: \n", correspondences)

		y0 = y0[correspondences[0].astype(int)]
		# print("y0: \n", y0, np.shape(y0))
		# print("y: \n", y, np.shape(y))

		#reshape Ys to be [ _ , 1] 
		y_reshape = np.reshape(y, (np.shape(y)[0]*2,1), order='C')
		y0_reshape = np.reshape(y0, (np.shape(y0)[0]*2,1), order='C')
		# print("y0_reshape: \n", np.shape(y0_reshape))

		#reorder U and L according to correspondences
		#	NOTE: here the subscript _i refers to the fact that this is the COMPLETE vector at cycle i
		U_i = U[correspondences[0].astype(int)] #this is straightforward for U
		
		#not as straightforward for L
		L_i = np.zeros(np.shape(correspondences)[1]*2)
		for ct in range(np.shape(correspondences)[1]):
			# print("ct ", ct, " corr ", correspondences[0,ct])
			L_i[2*ct:(2*ct+2)] = L[(2*correspondences[0,ct].astype(int)):(2*correspondences[0,ct].astype(int)+2)]

		# print("correspondences ", correspondences[0].astype(int), np.shape(correspondences))
		print("L ",L, np.shape(L))
		# print("L_i",L_i, np.shape(L_i))
		# print("U_i",U_i, np.shape(U_i))


		#get weighting matrix from covariance matrix
		W = get_weighting_matrix(cov2, npts2, U = U_i, L = L_i)
		# W = np.identity(np.shape(ctr2)[0]*2) #debug: simple identity for W
		# print("W[:4,:4] = \n", W[:4,:4])

		#create z which ignores extended directions of base scan ellipses
		# z0 = 

		H = get_H(y_reshape, x)
		H_w = weighted_psudoinverse(H, W)

		#TODO -> replace dy to remove elongated directions ---------------
		dy = y_reshape - y0_reshape
		dx = H_w.dot(dy)
		x -= dx
		# print("dx = ", dx)

		#incrementally update y
		# print("y: ", np.shape(y), " y init: ", np.shape(y_init))
		rot = R(x[2])
		t = x[0:2]
		y = rot.dot(y_init.T) + t
		y = y.T
		# print("y new", np.shape(y))
		P_corrected = rot.dot(pp2.T) + t
		P_corrected = P_corrected.T

		y0 = y0_init
		print("error: \n", np.sum(abs(y_reshape- y0_reshape))) #----------

		#draw progression of centers of ellipses
		# ax.plot(y.T[0,:], y.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1.), ls = '', marker = '.', markersize = 10)
		
		# #draw all points progressing through transformation
		# ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),0.025), ls = '', marker = '.', markersize = 20)

	#draw final translated points (for debug)
	# ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1,0,0,0.0375), ls = '', marker = '.', markersize = 20)

	#draw final translated points using initial P and final X
	rot = R(x[2])
	t = x[:2]
	P_final = rot.dot(pp2.T) + t
	ax.plot(P_final.T[:,0], P_final.T[:,1], color = (1,0,0,0.0375), ls = '', marker = '.', markersize = 20)

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

def get_state_error():

	P = None #TODO

	return P 

