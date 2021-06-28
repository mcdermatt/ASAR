import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import misc
from utils import *

#TODO:
#	account for overlapping grid cells

def NDT(Q,P,fig,ax, fid = 10, num_cycles = 1, draw = True):

	"""from Peter Biber, 2003
	
		fid -> breaks map down into fid x fid pieces """

	# 1) Build the NDT of the first scan.
	# 2) Initialize the estimate for the parameters (by zero or
	# 		by using odometry data).
	# 3) For each sample of the second scan: Map the
	# 		reconstructed 2D point into the coordinate frame of
	# 		the first scan according to the parameters.
	# 4) Determine the corresponding normal distributions
	# 		for each mapped point.
	# 5) The score for the parameters is determined by
	# 		evaluating the distribution for each mapped point
	# 		and summing the result.
	# 6) Calculate a new parameter estimate by trying to
	# 		optimize the score. This is done by performing one
	# 		step of Newton’s Algorithm.
	# 7) Goto 3 until a convergence criterion is met.

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# speed things up by decimating points
	# print("before ", np.shape(P))
	# P = P[::10]
	# print("after ", np.shape(P))

	#get point positions in 2d space and draw 1st and 2nd scans
	pp1 = draw_scan(Q,fig,ax, pt = 2) # set as 2 to hide it so we can see ellipses
	pp2 = draw_scan(P,fig,ax, pt = 1) #pt number assigns color for plotting

	#	---------------------------------------------------------
	# 1) Build the NDT of the first scan.
	E1 = subdivide_scan(pp1,fig,ax, fidelity = fid, pt =0)

	#	----------------------------------------------------------
	# 2) Initialize the estimate for the parameters (by zero or
	# 		by using odometry data)
	x = np.zeros([3,1]) #[ t_x, t_y, theta] }we want to optimize a single euler angle rather than a matrix

	#get 2d coords of centers of each ellipse from E1
	ctr = np.zeros([len(E1),2])
	for idx, c in enumerate(E1):
		ctr[idx,:] = c[0]

	#debug
	ax.plot(ctr[:,0], ctr[:,1], 'ko')

	results = []

	for cycle in range(num_cycles):
		# print(" ------------------ cycle: ", cycle, "----------------------" )

		scores = np.zeros(4)
		grad = np.zeros([3,1])
		grad_step_xy = 0.1 #how far we step in x and y when calculating gradients
		grad_step_theta = 0.01
		stepsize = 0.0001 #how fast we move in gradient descent

		#compute score for baseline and differential changes in each parameter
		for param in range(4): 

			#get baseline
			x_temp = np.zeros(np.shape(x))
			x_temp[:]  = x[:] #x.copy() #Debug this...

			#differentially change x and y one at a time
			if param == 0:
				x_temp[0] += grad_step_xy
			if param == 1:
				x_temp[1] += grad_step_xy
			if param == 2:
				x_temp[2] += grad_step_theta

			# 	---------------------------------------------------------
			# 3) For each sample of the second scan: Map the
			# 		reconstructed 2D point into the coordinate frame of
			# 		the first scan according to the parameters.

			#apply transformation
			t = x_temp[:2]
			rot = R(x_temp[2]) 
			P = rot.dot(pp2.T) + t

			# if cycle == num_cycles - 1:
			# 	dc = True
			# else:
			# 	dc = False 
			dc = False
			correspondences = get_correspondence(P,ctr.T,fig,ax, draw = dc)
			# print("c ", correspondences, np.shape(correspondences))

			#	----------------------------------------------------------
			#4) Determine the corresponding normal distributions
			# 		for each mapped point.
			score = 0
			for index, i in enumerate(correspondences[0]): 

				mu = E1[int(i)][0]
				sigma = E1[int(i)][1]

				eig = np.linalg.eig(sigma)
				eigenval = eig[0]
				eigenvec = eig[1]

				#get major and minor axis length of corresponding error ellipse
				#debug - figure out which is which
				major = np.sqrt(eigenval[1]) + 3 #debug: add minimum size to each ellipse (without this errors explode...)
				minor = np.sqrt(eigenval[0]) + 3

				#get rotation of ellipse
				theta_temp = np.arcsin(eigenvec[0,1]/eigenvec[0,0])

				#rotate point about origin so that axis of ellipse can be aligned with x,y axis
				rot_temp = R(theta_temp)
				# pt_rot = rot_temp.dot(pp2[index]) #DEBUG -THIS IS THE PROBLEM??!!!
				pt_rot = rot_temp.dot(P[:,index])

				#figure out how much I need to scale 1STD ellipse to reach point
				ratio = major/minor			
				b = np.sqrt( (pt_rot[0]**2)/(ratio**2) + pt_rot[1]**2 )
				a = ratio*b

				#use z_score as error
				z_score = (a / major)**2 #number of standard deviations point is from closest ellipse center
				score += z_score #keep track of overall scores

			#	-----------------------------------------------------------------
			# 5) The score for the parameters is determined by
			# 		evaluating the distribution for each mapped point
			# 		and summing the result.

			# print(" score: ", score, "param ", param)# "x_temp ", x_temp)
			scores[param] = score

		#	-----------------------------------------------------------------
		#6) Calculate a new parameter estimate by trying to
		# 		optimize the score. This is done by performing one
		# 		step of Newton’s Algorithm.

		# print("score: ",scores[-1])
		results = np.append(results, score)

		#calculate gradients
		grad[0] = (scores[0] - scores[3])
		grad[1] = (scores[1] - scores[3])
		grad[2] = (scores[2] - scores[3])

		#make sure no gradients are 0
		# grad[ grad < 0.01] = np.random.randn()*0.1

		#normalize
		# grad = grad/ np.sum(grad)
		# print("grad ", grad.T)
		
		#simple
		# dx = stepsize * np.squeeze(grad) * np.array([grad_step_xy, grad_step_xy, grad_step_theta]) 

		#true Newton
		dx = np.zeros(3)
		dx[0] = stepsize*scores[0] / (scores[0] - scores[3]) 
		dx[1] = stepsize*scores[1] / (scores[1] - scores[3]) 
		dx[2] = stepsize*scores[2] / (scores[2] - scores[3]) 


		# print("x", x.T)
		# print("dx ",dx)

		x -= dx[:,None]

	#draw transformed point set
	rot_final = R(x[2])
	t_final = x[:2]

	P_corrected = rot_final.dot(pp2.T) + t_final
	ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.0625), ls = '', marker = '.', markersize = 15)
		
	return x[2], t_final, results
