import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import misc
from utils import *
from NDT import NDT
from ICP import ICP_least_squares

#TODO:
# 		Conditioning: Remove the directions of the equation associated with the removed eigenvalues in the
# 			previous step; solve only the equations for which the condition number is greater than 1.

#		which eigenvectors/ eigenvals go with which directions in x,y,theta???

#		covariance of deltay is sum of individual covariances (see page 7)

#		efficient test for existance of extended direction (see page 14)

#		I am still creating a full weighting matrix when computing the 
#			Idea: defaut to doing fast weighted psudoinverse method, if things start to
#					explode then try using the full W method w/ condition number

#		Account for noise covariance of 2nd scan

#Notes:


def get_condition(H_w1):

	"""Condition number: ratio of largest to smallest singular values
		singular values are eigenvalues of (A.T).dot(A)
		the bigger the condition number the more singular the matrix 
	
		inputs: H_w1 -> term to be inverted

		outputs: L2 = used to remove axis with singular directions
				 lam = identity * eigenvalues of input


	1- Starting with (H’*W*H) prior to inverting it 
	
	2- Doing the eigen transform to find U (eigenvectors) and the associated eigenvalues Λ
	
	3- Start with the set of all eigenvalues and determine the condition number as the ratio of
	the largest-to-smallest eigenvalue. If the condition number is above a cutoff (say 10^6)
	then remove the lowest eigenvalue from the set. Repeat until the set of remaining
	eigenvalues has a condition number below the cutoff. (Unless the equations are trivial,
	the final set will have at least one eigenvalue left, in which case the condition number is
	one.)
	
	4- Remove the directions of the equation associated with the removed eigenvalues in the
	previous step; solve only the equations for which the condition number is greater than 1.

		"""

	cutoff = 10e5 #10e6

	#think of H_w1 as an ellipse- we are looking for SHORT principal axis
	# print("H _w1: \n", H_w1)

	eig = np.linalg.eig(H_w1)
	eigenval = eig[0]
	eigenvec = eig[1] 
	# print("eigenval: \n ", eigenval) #there are 3 values here
	# print("eigenvectors: \n", eigenvec)

	small, middle, big = sorted(eigenval)[0:3]
	# print(small,middle,big)

	condition = abs(big / small)
	# print(condition < cutoff)

	#knock off lowest axis until condition is less than cutoff
	if condition > cutoff:
		condition = big/ middle
		# print(condition < cutoff)

		#get axis for eigenvector cooresponding to small eigeval
		remainingaxis1 = np.argwhere(eigenval != small)
		# print("r1", remainingaxis1)
		# L2 = eigenvec[remainingaxis1] #was this
		L2 = np.identity(3)[remainingaxis1]

		if condition > cutoff:
			condition = big
			# print(condition < cutoff)

			remainingaxis2 = np.argwhere(eigenval != middle)
			# print("r2", remainingaxis2)
			# L2 = eigenvec[np.intersect1d(remainingaxis1, remainingaxis2)] #was this
			L2 = np.identity(3)[np.intersect1d(remainingaxis1, remainingaxis2)]

			#fix shape to make 2d
			# L2 = L2[:,None]

			if condition > cutoff:
				# H_w1 = np.identity(1) <- does not work?
				# condition = 1
				L2 = np.zeros([3,3])
				# L2 = np.identity(3)
				print("TODO: fix this case")

		if len(np.shape(L2)) > 2:
			L2 = np.squeeze(L2)
		H_w1 = L2.dot(H_w1)

	else:
		L2 = np.identity(3)

	# lam = eigenval[:,None]
	lam = np.identity(3)
	for m in range(3):
		lam[m,m] = eigenval[m]

	U2 = eigenvec

	# print("L2: \n", L2)
	# print("lam: \n", lam)
	# print("condition: \n", condition)

	return L2, lam, U2

def visualize_U(U, L, ctr1, cov1, fig, ax):

	"""view the effects of U on the major and minor axis of ellipses"""

	for i in range(np.shape(U)[0]):

		eig = np.linalg.eig(cov1[i])
		eigenval = eig[0]

		#get minor axis
		minor_radius = np.array([np.sqrt(min(eigenval))*2, 0]) #np.array([10,0])
		# ax.plot([ctr1[i,0], minor_stop[0]], [ctr1[i,1], minor_stop[1]], 'k-' , lw = 2)
		minor_stop = minor_radius.dot(U[i])

		#get major axis
		U_alt = R_alt(U[i])
		major_radius = np.array([np.sqrt(max(eigenval))*2, 0]) #np.array([10,0])
		major_stop = major_radius.dot(U_alt)
		# major_stop = major_radius.dot(U[i])

		#draw
		hw = 0 #arrow head width
		#case 1: no extended axis
		if np.all(L[i] == np.identity(2)):
			ax.arrow(ctr1[i,0],  ctr1[i,1], minor_stop[0], -minor_stop[1], length_includes_head = True, lw = 1, head_width = hw)
			ax.arrow(ctr1[i,0], ctr1[i,1], major_stop[0], -major_stop[1], length_includes_head = True, lw = 1, head_width = hw)
			ax.arrow(ctr1[i,0],  ctr1[i,1], -minor_stop[0], minor_stop[1], length_includes_head = True, lw = 1, head_width = hw) #repeat
			ax.arrow(ctr1[i,0], ctr1[i,1], -major_stop[0], major_stop[1], length_includes_head = True, lw = 1, head_width = hw)

		if np.all(L[i] == np.array([[1,0]])):
			ax.arrow(ctr1[i,0],  ctr1[i,1], -minor_stop[0], minor_stop[1], length_includes_head = True, lw = 1, head_width = hw)
			ax.arrow(ctr1[i,0],  ctr1[i,1], minor_stop[0], -minor_stop[1], length_includes_head = True, lw = 1, head_width = hw)

		if np.all(L[i] == np.array([[0,1]])):
			ax.arrow(ctr1[i,0], ctr1[i,1], -major_stop[0], major_stop[1], length_includes_head = True, lw = 1, head_width = hw)
			ax.arrow(ctr1[i,0], ctr1[i,1], major_stop[0], -major_stop[1], length_includes_head = True, lw = 1, head_width = hw)
			# print("special case")

def get_U_and_L(cov1, cellsize = np.array([100,100])):

	"""generates matrices used to remove axis in which ellipses for each voxel are extended 
			overly extended axis is defined as:
			ax > (L^2)/16, where L is the length of the voxel 
		
		inptus: cov1 -> cov from ellipse data from first scan taken from subdivide_scan()[1]

		outputs: FOR EACH VOXEL 
				 U -> rotation matrix to align point with the major and minor axis of ellipse
				 L -> reduced dimension array used to ignore extended axis directions
	
			"""	
	# cellsize = cellsize*0.9 #to be a little extra conservative
	# print("Using cellsize = ", cellsize)

	U = np.zeros([np.shape(cov1)[0],2,2])

	# don't know what size L is going to be before we start 
	L = []

	#loop through every voxel in scan 1 (only do this once per keyframe)
	for i in range(np.shape(cov1)[0]):

		eig = np.linalg.eig(cov1[i])
		eigenval = eig[0]
		eigenvec = eig[1]

		#get new coordinate frame
		# theta_temp = np.arctan(eigenvec[0,1]/eigenvec[0,0]) #was this
		# if eigenvec[0,1] < eigenvec[0,0]:
			# theta_temp += np.pi/2
		theta_temp = np.arctan2(eigenvec[0,1],eigenvec[0,0]) #test

		# print(theta_temp)

		#get U matrix requred to rotate future P points so that x', y' axis align with major and minor axis of ellipse
		# print("eigenvec", eigenvec)
		if np.cos(theta_temp) > 0:
			U[i] = eigenvec.dot(R(np.pi/2)) #if not rotated past 45 deg
		else:
			U[i] = -eigenvec #if rotated past 45 deg

		# U[i] = -eigenvec #if not rotated past 45 deg
		# U[i] = eigenvec.dot(R(np.pi/2)) #if rotated past 45 deg


		# get L matrix 
		# NOTE: axis1 is not always the bigger or smaller axis...
		# axis1 = np.sqrt(eigenval[0])
		# axis2 = np.sqrt(eigenval[1])

		# print(eigenval)

		#test
		axis1 = np.sqrt(max(eigenval))
		axis2 = np.sqrt(min(eigenval))

		# print("axis1: ", axis1, " axis2: ", axis2)

		#get projections of major and minor axis in x and y directions
		axis1x = abs(np.cos(theta_temp)*axis1)
		axis1y = abs(np.sin(theta_temp)*axis1)
		axis2x = abs(np.cos(theta_temp + np.pi/2)*axis2)
		axis2y = abs(np.sin(theta_temp + np.pi/2)*axis2)

		#base case: no elongated directions
		if axis1x<(cellsize[0]/4) and axis1y<(cellsize[1]/4) and axis2x<(cellsize[0]/4) and axis2y<(cellsize[1]/4):
			L_i = np.array([[1,0],[0,1]])
		#elongated axis1
		if (axis1x>(cellsize[0]/4) or axis1y>(cellsize[1]/4)):
			if (axis2x<(cellsize[0]/4) and axis2y<(cellsize[1]/4)):
				L_i = np.array([[1,0]])
			else:
				#both elongated
				L_i = np.zeros([1,2])
		# 		print("problem here")


		#elongated axis2
		if (axis2x>(cellsize[0]/4) or axis2y>(cellsize[1]/4)):
			if (axis1x<(cellsize[0]/4) and axis1y<(cellsize[1]/4)):
				L_i = np.array([[0,1]])

		# L = np.append(L, L_i, axis = 0)
		L.append(L_i)

		# print("L_i for this step \n", L_i)

		#TODO- remove axis that are all zeros(?) 
		#		not sure if I need this...

	# L = L[:,None]

	#NOTE:
	#	U and L are in relation to the origonal scan, but the coorespondences used in the main loop 
	#	will be in a different order


	return U, L

def get_weighting_matrix(cov1, npts1, cov2, npts2, L, U):

	'''
	cov: 3D matrix containing covariance matrices for all voxels 
	
	npts: number of points inside each voxel of cov

	R_noise: sensor noise model, (should reflect the spread of points due to ______)
		R = Q /(|J|), Q = true covariance of voxel J, |J| = # pts in J
			size(R) = [J,J] where J is the total number of voxels

	'''

	W = np.identity(np.shape(cov2)[0]*2) #multiply by 2 because cov matrices are 2x2


	# print("U \n", np.shape(U))
	# print("L \n", L, np.shape(L))

	for i in range(np.shape(cov2)[0]): 

		#normalize true covariance by the number of points in the subdivision
		R_noise = (cov1[i] / (npts1[i] - 1)) + (cov2[i] / (npts2[i] - 1))  
		# R_noise = (cov2[i] / (npts2[i] - 1))  
		

		#NOTE: underscript _j denotes that this is the jth voxel in the scan
		# L_j = L[2*i:(2*i+2)][:,None].T #deprecated
		# L_j = L[i]
		# U_j = U[i]

		# U_TRU = U_j.T.dot(R_noise.dot(U_j))
		# # print("U_TRU", np.shape(U_TRU))
		# LU_TRU = L_j.dot(U_TRU)
		# # print("LU_TRU", np.shape(LU_TRU))
		# R_z = (L_j.T).dot(LU_TRU)
		R_noise = np.linalg.multi_dot((L[i], U[i].T, R_noise, U[i], L[i].T))

		# print("R_z = ", R_z)

		#test
		W[(2*i):(2*i+2),(2*i):(2*i+2)] = np.linalg.pinv(R_noise)

		#works well without using U and L
		# W[(2*i):(2*i+2),(2*i):(2*i+2)] = np.linalg.pinv(R_noise)


	# print(np.floor(R_noise[:12,:12])) #make sure everything looks like the right shape

	# W = np.linalg.inv(R_noise) #does not work (singular matrix error)
	# W = np.linalg.pinv(R_noise) # also changed to pinv in weighted_psudoinverse()

	# W = np.linalg.pinv(W)
	# print("W: \n", np.shape(W))
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

def get_dx(y, y0, x, cov1, cov2, npts1, npts2, L, U):

	'''gets dx accounting for condition number. this version requires construction of full W 

		inputs: y 	-> centers of ellipses of 2ND SCAN [2N , 1]
				X 	-> [x y theta]
				npts-> 1D array containing the number of points in each ellipse
				L  	->  L for voxel i
				U 	-> U for voxel i 

		outputs: dx 

			   TAKE INTO ACCOUNT MAIN L AND U -> is this being done in getting Weighting matrix
		'''

	H = get_H(y,x)
	#slow way of getting HTWH -------------------
	# W = get_weighting_matrix(cov, npts, L, U)
	# # print("W =  \n", W[:6,:6])
	# HTWH = H.T.dot(W).dot(H)
	# print("HTWH (from get_dx) = \n", HTWH)
	#--------------------------------------------

	#fast way of getting HTWH--------------------
	HTWH = np.zeros([3,3])
	HTWH_alt = np.zeros([3,3]) #need this while I debug because there is a sign error somewhere
	HTW = np.zeros([3, np.shape(y)[0]])
	Q = np.zeros([3,3])
	for i in range(np.shape(y)[0]//2): #loop through each usable voxel
		#estimate R_noise for voxel j
		R_noise = (cov1[i] / (npts1[i] - 1)) + (cov2[i] / (npts2[i] - 1))
		R_noise = np.linalg.multi_dot((L[i], U[i].T, R_noise, U[i], L[i].T)) # was this
		# R_noise = np.linalg.multi_dot((L[i], U[i], R_noise, U[i].T, L[i].T)) #test
		#calclate voxel j's contribution to the first term of H_w -------------------------
		# H_w1  = [ H_wj1[0] + H_wj1[1] + H_wj1[2] + ... ] <- should be a 3x3 matrix
		# H_wj1 = (H.T)(W)(H) = (H.T)(R^-1)(H)
		H_z = L[i].dot(U[i].T).dot(H[2*i:2*i+2])
		HTWH_j = H_z.T.dot(np.linalg.pinv(R_noise)).dot(H_z)
		#add contributions of j
		HTWH += HTWH_j

		HTW_j = H_z.T.dot(np.linalg.pinv(R_noise))
		HTW[:,2*i:2*i+2] = HTW_j

		#update Q matrix (P in paper) ---------------------------------------
		# https://www.sciencedirect.com/topics/engineering/error-covariance-matrix
		# print(H[2*i:2*i+2])
		# print("L_i: ",L[i])
		# print("U_i: ", U[i])
		H_alt = H[2*i:2*i+2]
		H_alt[0,0] = -1
		H_alt[1,1] = -1
		R_noise_alt = np.ones([2,2])
		R_noise_alt = np.linalg.multi_dot((L[i], U[i].T, R_noise_alt, U[i], L[i].T))
		H_z_alt = L[i].dot(U[i].T).dot(H_alt)
		HTWH_j_alt = H_z_alt.T.dot(np.linalg.pinv(R_noise_alt)).dot(H_z_alt)
		HTWH_alt += HTWH_j_alt

	#CHECK CONDITION
	L2, lam, U2 = get_condition(HTWH)
	# print("L2: \n", L2) # -> removes axis
	# print("lam: \n", lam) # -> 3x3 diagonal array with eigenvalues of HTWH
	# print("U2: \n", U2) # -> COLUMNS are eigenvecs
	
	#pythonic way------------------------------
	#z_k = (1/lam_k)[(U.T)(H.T)(W)(y)]_k
	#z_k = ([U.T]_k)(x)

	#TODO- account for L by removing extended axis from lam
	# print("H.T \n", np.shape(H.T))

	# dy = y - y0

	# dz = np.zeros([3,1])
	# for k in range(3):
	# 	dz[k] = (1/lam[k,k])*( (U2.T).dot(H.T).dot(W).dot(dy)[k] )

	# dx = np.linalg.pinv(U2.T).dot(dz)
	#------------------------------------------

	#linalg way--------------------------------
	# (L)(U2.T)(H.T)(W)(y-y0) = (L)(lam)(U.T)(x)
	# (z) = (U.T)(x)

	dy = y - y0

	#works
	dx = np.linalg.pinv( L2.dot(lam).dot(U2.T) ).dot(L2).dot(U2.T).dot(HTW).dot(dy)

	# with no L2 pruning (will explode in geometrically ambiguous situations)
	# dx = np.linalg.pinv(HTWH).dot(HTW).dot(dy)
	
	#------------------------------------------

	# without pruning from L2
	Q = np.linalg.pinv(HTWH) #CORRECT FOR OUTPUT COVARIANCE MATRIX
	# Q = np.linalg.pinv(HTWH_alt)

	#dumb way that might work
	# Q = dx.dot(np.linalg.pinv(HTW.dot(dy)))

	return dx, Q

def fast_weighted_psudoinverse(y, x, cov1, npts1, cov2, npts2, L, U):

	'''returns array H_w without creating full square matrix for W
	
		inputs: y 	-> centers of ellipses of 2ND SCAN [2N , 1]
				X 	-> [x y theta]
				npts-> 1D array containing the number of points in each ellipse
				L  	->  L for voxel i
				U 	-> U for voxel i 

		outputs: H_w -> psudoinverse

	Question: Do I truncate parts of scan 2 because the matching axis on scan 1 are too far extended?

	'''
	#OLD --------------------------------------------------------------------------------------------------------
	# truncate L matrix to remove all elements in directions of extended baseline error ellipses
	# we want to do this in here so we have the indices of the axis to be ignored so we can do the same to R(?)
	
	# nonzero_elements = np.argwhere(L[:] != np.array([0,0]))[:,0]
	# L_i_truncated = L[nonzero_elements]
	# # print("L_i truncated \n",np.shape(L_i_truncated))

	# y_truncated = y[nonzero_elements]
	# y = y_truncated

	# print("y \n", y, "\n y_truncated \n", y_truncated)

	# ------------------------------------------------------------------------------------------------------------

	H = get_H(y,x)
	# print("H", np.shape(H))

	#init 1st and 2nd terms
	H_w1 = np.zeros([3,3])
	H_w2 = np.zeros([3,np.shape(y)[0]])

	for i in range(np.shape(y)[0]//2):

		#estimate R_noise for voxel j
		# R_noise = cov2[i] / (npts2[i]-1)
		R_noise = (cov1[i] / (npts1[i] - 1)) + (cov2[i] / (npts2[i] - 1)) 	#ignoring U and L
		# R_noise = U[i].dot(cov[i]).dot(U[i].T) / (npts[i] - 1) 	#need to account for rotation

		#adjust R_noise to account for the effects of L and U ------------------------
		# R_z = (L)(U.T)(R)(U)(L.T)
		#	  should remain [2x2] matrix

		#DEBUG: comment out line below to ignore U, L
		R_noise = np.linalg.multi_dot((L[i], U[i].T, R_noise, U[i], L[i].T))
		# print("R_noise: ", np.shape(R_noise)) # should be 2x2
		# print("L[i]", L[i])
		# print("L[i].dot(U[i].T)", L[i].dot(U[i].T))

		#calclate voxel j's contribution to the first term of H_w -------------------------
		# H_w1  = [ H_wj1[0] + H_wj1[1] + H_wj1[2] + ... ] <- should be a 3x3 matrix
		# H_wj1 = (H.T)(W)(H) = (H.T)(R^-1)(H)

		#was this
		# H_wj1 = H[2*i:2*i+2].T.dot(np.linalg.pinv(R_noise)).dot(H[2*i:2*i+2]) 
		#trying this
		H_z = L[i].dot(U[i].T).dot(H[2*i:2*i+2])
		H_wj1 = H_z.T.dot(np.linalg.pinv(R_noise)).dot(H_z)

		# cond_j = get_condition(H_wj1)
		# print(cond_j < 10e8)

		#add contributions of j to first term
		H_w1 += H_wj1

		#calculate voxel j's contributuion to the 2nd term of H_w -------------------------
		# H_w2 = [ H_wj2[0]  H_wj2[1]  H_wj2[2]  ... ]  <- should be a 3xN matrix, N = #pts
		# H_wj2 = (H.T)(R^-1)

		#was this
		# H_wj2 = H[2*i:2*i+2].T.dot(np.linalg.pinv(R_noise)) 
		#trying this
		H_wj2 = H_z.T.dot(np.linalg.pinv(R_noise))

		#assign H_wj2 to position in in H_w2
		H_w2[:,2*i:2*i+2] = H_wj2

		# print("H_wj1", H_wj1)
		# print("H_wj2", H_wj2)

	# print("H_w1", np.shape(H_w1)) #3x3
	# print("H_w2", np.shape(H_w2)) #3xN

	#CHECK CONDITION TO MAKE SURE FIRST TERM IS INVERTABLE - if not this will correct it
	# print("H_w1 (from fast_weighted_psudoinverse) = \n", H_w1)
	L2, lam, U2 = get_condition(H_w1)
	# print("L2 (from fast_weighted_psudoinverse)= ", L2, np.shape(L2))
	# print("lam = ", lam)
	#NOTE:
	#	we now want to remove all short axis of H'WH so that this first term is invertable

	#compute dot product of the two terms -------------------------------------
	# does not take into account L2 which removes axis of first term to make it invertable:
	# H_w = np.linalg.pinv(H_w1).dot(H_w2) 

	# with L2:
	H_w = np.linalg.pinv(L2.dot(H_w1)).dot(L2.dot(H_w2)) #was this
	# H_w = np.linalg.pinv(L2.dot(lam).dot(H_w1)).dot(L2.dot(H_w2)) #need to account for lambda? 
	#		this is the dimensionality problem
	#--------------------------------------------------------------------------

	# print(np.shape(H_w))
	# print(np.argwhere(H_w == 0))

	return H_w


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

def ICET_v2(Q,P,fig,ax,fid = 10, num_cycles = 1, min_num_pts = 5, draw = True, along_track_demo = False):

	"""similar to v1 except uses "weighted" psudoinverse instead of LSICP"""

	if along_track_demo == True:

		pp1, pp2, x_actual = generate_along_track_data(fig,ax, draw = True, output_actual = True)

	else:
		#get point positions in 2d space and draw 1st and 2nd scans
		pp1 = draw_scan(Q,fig,ax, pt = 2)
		pp2 = draw_scan(P,fig,ax, pt = 2) #pt number 0,1 assigns color for plotting, pt = 2 does not draw

	#get cellsize param for calculating L
	minx = np.min(pp1[:,0])
	maxx = np.max(pp1[:,0])
	miny = np.min(pp1[:,1])
	maxy = np.max(pp1[:,1])

	cy = (maxy - miny) / (fid) #cellsize x
	cx = (maxx - minx) / (fid) #cellsize y
	cellsize = np.array([cx, cy]) #/ 2
	# cellsize = (cx+cy)/2
	# cellsize = 10

	#group points into ellipses
	E1, pp1_lims = subdivide_scan(pp1,fig,ax, fidelity = fid, pt = 0, 
		min_num_pts = min_num_pts, output_lims = True)
	_ = subdivide_scan(pp2,fig,ax, fidelity = fid, pt = 1, 
		min_num_pts = min_num_pts, lims = pp1_lims) #added back for viz

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
	U, L = get_U_and_L(cov1, cellsize)

	#make sure U is being calculated correctly ---------------------------------------
	# print(np.shape(U))
	visualize_U(U, L, ctr1, cov1, fig, ax)
	#---------------------------------------------------------------------------------

	# print("L \n", L, np.shape(L))

	#inital estimate for transformation
	#TODO: improve initial estimated transform (start with zeros here, could potentially use wheel odometry) 
	x = np.zeros([3,1])
	best_x = np.zeros([3,1])

	y0 = ctr1
	y0_init = ctr1 #hold on to initial centers so we don't lose information when doing correspondences
	P_corrected = pp2 #for debug

	error = np.zeros(num_cycles)
	best_error = 10e6

	for cycle in range(num_cycles):

		# print("cycle ", cycle, " --------------------")

		#Refit 2nd scan into voxels on each iteration
		E2 = subdivide_scan(P_corrected,fig,ax, fidelity = fid, pt = 2, 
				min_num_pts = min_num_pts, lims = pp1_lims)

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

		#reshape Ys to be [ 2N , 1] 
		y_reshape = np.reshape(y, (np.shape(y)[0]*2,1), order='C')
		y0_reshape = np.reshape(y0, (np.shape(y0)[0]*2,1), order='C')
		# print("y0_reshape: \n", np.shape(y0_reshape))

		#reorder U and L according to correspondences
		#	NOTE: here the subscript _i refers to the fact that this is the COMPLETE vector at cycle i
		U_i = U[correspondences[0].astype(int)] #this is straightforward for U

		#NEW - when L is truncated for cases with extended axis ------------------------------------
		# print(correspondences[0].astype(int))
		#rearrange L_i to be in order of correspondances
		L_i = []
		for i in correspondences[0].astype(int):
			L_i.append(L[i])
		#-------------------------------------------------------------------------------------------

		#needs debug...
		# reorder cov1 and npts1 according to correspondences
		# print(np.shape(npts1), np.shape(npts2))
		# need to get reverse correspondences for cov1 and npts1
		# reverse_corr = get_correspondence(y0_init.T, y.T,fig,ax,draw=False)[0].astype(int)
		# print("correspondences: \n", reverse_corr, np.shape(reverse_corr))
		# print("cov1: \n", np.shape(cov1))
		# cov1 = cov1[reverse_corr]
		# npts1 = npts1[reverse_corr]

		#create array the same size as cov2 that holds the covariance matrices from cov1 that 
		#	are associated with are nearest point accoring to the correspondance array
		cov1reorder = np.zeros(np.shape(cov2))
		npts1reorder = np.zeros(np.shape(npts2))
		for c in range(np.shape(cov1reorder)[0]):
			cov1reorder[c] = cov1[correspondences[0].astype(int)[c]]
			npts1reorder[c] = npts1[correspondences[0].astype(int)[c]]
		# print(cov1reorder[c])

		# "standard" weighted psudoinverse ------------------------------------
		#get weighting matrix from covariance matrix 
		# Using standard funcs with full block diagonal matrix for W
		# W = get_weighting_matrix(cov2, npts2, L_i, U_i)
		# W = np.identity(np.shape(ctr2)[0]*2) #debug: simple identity for W
		# print("W[:4,:4] = \n", W[:4,:4])
		# H = get_H(y_reshape, x)
		# H_w = weighted_psudoinverse(H, W)
		# ------------------------------------------------------------------

		# Fast weighted psudoinverse ---------------------------------------
		H_w = fast_weighted_psudoinverse(y_reshape, x, cov1reorder, npts1reorder, cov2, npts2, L_i, U_i)
		# print("H_w: \n", H_w, np.shape(H_w))
		# print("Q_test = ", H_w.dot(cov1).dot((H_w.T)))
		#-------------------------------------------------------------------

		#NOTE 7/21: I think I should be using L and lambda OUTSIDE fast_weighted_psudoinverse()
		dxTest, Q = get_dx(y_reshape, y0_reshape, x, cov1reorder, cov2, npts1reorder, npts2, L_i, U_i)
		# print("dxTest = \n", dxTest)
		#TODO: I'm getting different HTWH values when using fast weighted psudoinverse and getdx
		# print(Q)
		#-------------------------------------------------------------------

		z = np.zeros(np.shape(y_reshape))
		z0 = np.zeros(np.shape(y0_reshape))
		for count in range(np.shape(U_i)[0]//2):
			#y_hat = (U.T).dot(y_reshape) <- need to loop through U's to get rotation
			#	NOTE: U_i[count].T.dot(...) == np.linalg.pinv(U_i[count].T).dot(...)

			z[2*count:2*count+2] = U_i[count].T.dot(y_reshape[2*count:2*count+2]) #these were using U instead of U_i...
			z0[2*count:2*count+2] = U_i[count].T.dot(y0_reshape[2*count:2*count+2])

		dz = z - z0
		dx = H_w.dot(dz)
		# dy = y_reshape - y0_reshape
		# dx = H_w.dot(dy) #for no U,L		
		# x -= dx #from fast weighted psudoinverse
		x -= dxTest #from get dx

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

		error[cycle] = np.sum(abs(y_reshape- y0_reshape))
		# print(y_reshape[:4].T, "\n", y0_reshape[:4].T, "\n", y_reshape[:4].T-y0_reshape[:4].T)

		#save best transformation
		if error[cycle] < best_error:
			best_error = error[cycle]
			best_x[:] = x[:]
			# print(best_error)
			# print(best_x)
		#draw progression of centers of ellipses
		# ax.plot(y.T[0,:], y.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),0.25), ls = '', marker = '.', markersize = 10)
		
		# #draw all points progressing through transformation
		# ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),0.025), ls = '', marker = '.', markersize = 20)

	#draw final translated points (for debug)
	# ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1,0,0,0.0375), ls = '', marker = '.', markersize = 20)

	#draw final translated points using initial P and final X
	rot = R(best_x[2])
	t = best_x[:2]
	P_final = rot.dot(pp2.T) + t
	ax.plot(P_final.T[:,0], P_final.T[:,1], color = (1,0,0,0.0375), ls = '', marker = '.', markersize = 20)

	if along_track_demo == True:
		return best_x, Q, error, x_actual
	else:
		return best_x, Q, error

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

