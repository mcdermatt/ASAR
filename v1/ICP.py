import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import misc
from utils import *


def vanilla_ICP(Q,P, fig, ax, num_cycles = 3, draw = True):

	'''Iterative Closest Point algorithm using SVD
		find R, t such that: 
			R.P + t = Q

	Inputs:
	Q = standard deviation ellipses from 1st scan
			(what we want P to look like)
	P = standard deviation ellipses from 2nd scan
			(we want to translate and rotate this to look like Q)

	Outputs:
	R = rotation
	t = translation '''

	#	https://nbviewer.jupyter.org/github/niosus/notebooks/blob/master/icp.ipynb 


	true_data = Q
	moved_data = P
	P_corrected = moved_data

	center_q = true_data.mean(axis=1)
	print("center_q ", center_q)
	centered_true = true_data.T - center_q
	# ax.plot(centered_true[:,0],centered_true[:,1],'r-.') #draw Q recentered at zero ---------
	centered_true = centered_true.T

	for _ in range(num_cycles):

		#center data
		#TODO- account for outliars here (ignore poitns > nstd when computing mean)
		center_p = moved_data.mean(axis=1)
		print("center_p ", center_p) 	#debug: centers not changing over time?

		moved_data = moved_data.T - center_p
		# ax.plot(moved_data[:,0],moved_data[:,1],'b-.') #draw moved data -------------------------
		moved_data = moved_data.T

		moved_correspondence = get_correspondence(moved_data, centered_true, fig, ax, draw = draw)


		#calculate cross covariance: describes how a point in P will change along with changes in Q
		cov = get_cross_cov(moved_data, centered_true, moved_correspondence)

		#TODO: replace with least squares approach
		#~~~
		#get R and t from cross covariance 
		U, S, V_T = np.linalg.svd(cov) #TODO - compute this manually to review SVD
		R_found = U.dot(V_T)
		t_found = center_q - R_found.dot(center_p)

		#apply estimated R and t
		P_corrected = R_found.dot(P_corrected) + t_found[:,None] #TODO: fix this!!

		# print("PC ", P_corrected)
		# print("Squared diff: (P_corrected - Q) = ", np.linalg.norm(P_corrected - Q))
		#~~~

		moved_data = P_corrected

	ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.125), ls = '', marker = '.', markersize = 20)

	R = R_found
	t = t_found

	return R, t

def ICP_least_squares(Q,P, fig, ax, num_cycles = 1, draw = False):


	x = np.zeros([3,1])

	true_data   = Q
	P_corrected = P

	for cycle in range(num_cycles):

		H = np.zeros([3,3]) #Hessian 
		g = np.zeros([3,1]) #Gradient
		chi = 0

		if draw == True and cycle == num_cycles-1:
			draw_this_time = True
		else:
			draw_this_time = False

		correspondences = get_correspondence(P_corrected, true_data, fig, ax, draw = draw_this_time)

		#get H, g, chi
		for i in range(np.shape(correspondences)[1]):
			p =  P_corrected[:,i] #was this
			# p = P[:,i]	#debug: not this??
			q = true_data[:,int(correspondences[0,i])][:,None]

			err = error(x,p,q)
			weight = 1 #TODO: replace with lambda func at some point...

			J = jacobian(x, p)

			H += weight * J.T.dot(J)
			g += weight * J.T.dot(err)
			chi += err.T * err

		# print(H, g, chi)

		dx = np.linalg.lstsq(H, -g, rcond=None)[0] #TODO: recreate this func
		x += dx
		rot = R(x[2])
		t = x[0:2]
		x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2])) # normalize angle

		P_corrected = rot.dot(P_corrected) + t
		P_corrected = np.squeeze(P_corrected)

	ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.125), ls = '', marker = '.', markersize = 20)

	return P_corrected, t, rot


