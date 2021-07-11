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
	pp1 = draw_scan(Q,fig,ax, pt = 0) # set as 2 to hide it so we can see ellipses
	pp2 = draw_scan(P,fig,ax, pt = 1) #pt number assigns color for plotting

	P_corrected = pp2.T

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

	#debug- draw centers of ellipses from 1st scan
	ax.plot(ctr[:,0], ctr[:,1], 'ko')

	results = []

	for cycle in range(num_cycles):
		# print(" ------------------ cycle: ", cycle, "----------------------" )

		scores = np.zeros(4)
		grad_step_xy = 0.1 #how far we step in x and y when calculating gradients
		grad_step_theta = 0.01
		stepsize = 0.05 #how fast we move in gradient descent

		#compute score for baseline and differential changes in each parameter
		for param in range(4): 

			#get baseline
			x_temp = np.zeros(np.shape(x))
			x_temp[:]  = x[:]

			#differentially change x and y one at a time
			if param == 0:
				x_temp[0] += grad_step_xy
			if param == 1:
				x_temp[1] += grad_step_xy
			if param == 2:
				x_temp[2] += grad_step_theta
			# print("x_temp ",x_temp)

			# 	---------------------------------------------------------
			# 3) For each sample of the second scan: Map the
			# 		reconstructed 2D point into the coordinate frame of
			# 		the first scan according to the parameters.

			#apply transformation
			t_temp = x_temp[:2]
			rot_temp = R(x_temp[2]) 
			P_temp = rot_temp.dot(pp2.T) + t_temp #P_temp is P_corrected with differential changes to each param in x

			# 	---------------------------------------------------------
			# 3) For each sample of the second scan: Map the
			# 		reconstructed 2D point into the coordinate frame of
			# 		the first scan according to the parameters.

			#to debug correspondences
			# if (cycle == num_cycles - 1) and (param == 3):
			# 	dc = True
			# else:
			# 	dc = False 
			dc = False
			correspondences = get_correspondence(P_temp,ctr.T,fig,ax, draw = dc)
			# print("c ", correspondences[0], np.shape(correspondences))

			#	----------------------------------------------------------
			#4) Determine the corresponding normal distributions
			# 		for each mapped point.
			score = 0
			H = np.zeros([3,3])
			g = np.zeros([3,1])
			for index, i in enumerate(correspondences[0]): 

				mu = E1[int(i)][0][:,None]  	#center of corresponding point cloud from 1st scan (q_i in Biber paper)
				# print("mu", mu, np.shape(mu))
				# print("x_i", P_corrected[:,index])
				sigma = E1[int(i)][1]	#associated covariance matrix

				eig = np.linalg.eig(sigma)
				eigenval = eig[0]
				eigenvec = eig[1]

				#Biber paper: --------------------------------------------------------------
				# score += -e^( (q.T)(cov)(q) / 2 )
				# 		where: q = distance between x_i and corresponding NDT point  
				# 			   ** assume cov is positive definate

				q = P_temp[:,index][:,None] - mu
				E = np.linalg.pinv(sigma)
				score_i = -np.exp( (-(q).T.dot(E).dot(q) )/2 ) #----------------------------
						#NOTE: dealing with this type of score is SO FRUSTRATING				
				score += score_i

				#only needed in base case
				if param == 3:	

					#get jacobian
					J = jacobian(x, P_temp[:,index])

					#update gradient-----
					#for least squares
					# err = error(x,P_temp[:,index],mu)
					# g += J.T.dot(err)

					#From Biber paper
					# g += (q).T.dot(E).dot(J).T*score

					#update Hessian-----
					#for simple least squares
					# H += J.T.dot(J) #was this (does not account for score)

					#from Biber paper
					H += score_i*((-q.T.dot(E).dot(J)).dot((-q.T.dot(E).dot(J)).T))

					# 2nd_derivs = np.zeros([3,3,2]) #[i, j, ???]
					# 2nd_derivs[2,2,0] = -x[0]*np.cos(x[2]) + x[1]*np.sin(x[2])
					# 2nd_derivs[2,2,1] = -x[0]*np.sin(x[2]) - x[1]*np.cos(x[2])

			scores[param] = score

		# print(" score: ", score, "param ", param)# "x_temp ", x_temp)

		# calculate gradient
		g = np.zeros(3)
		g[0] = (scores[0] - scores[3])/grad_step_xy 
		g[1] = (scores[1] - scores[3])/grad_step_xy 
		g[2] = (scores[2] - scores[3])/ grad_step_theta

		print("g",g)
		print("H^-1", np.linalg.pinv(H))
		# print(np.all(np.linalg.eigvals(np.linalg.pinv(H)) > 0)) #tells us if positive definate
		
		# print("score: ",scores[-1])
		results = np.append(results, score)

		#	-----------------------------------------------------------------
		#6) Calculate a new parameter estimate by trying to
		# 		optimize the score. This is done by performing one
		# 		step of Newton’s Algorithm.
		#	(H)(delP) = -g
		#		H = hessian
		#		delP = how far we need to change parameters
		#		g = transposed gradient of f

		#Biber paper
		dx = np.linalg.pinv(H).dot(-g)
		#least squares
		# dx = np.linalg.lstsq(H, -g, rcond=None)[0]
		print("dx ",dx)

		x += dx[:,None]
		print("x = ",x)


		#DEBUG

		#draw all points progressing through transformation
		rot = R(x[2])
		t = x[0:2]
		P_corrected = rot.dot(pp2.T) + t
		P_corrected = P_corrected.T
		ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),0.025), ls = '', marker = '.', markersize = 20)


	#draw transformed point set
	rot_final = R(x[2])
	t_final = x[:2]

	P_corrected = rot_final.dot(pp2.T) + t_final
	ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.0625), ls = '', marker = '.', markersize = 15)
		
	return x[2], t_final, results




def NDT_old(Q,P,fig,ax, fid = 10, num_cycles = 1, draw = True):

	"""
	keeping this in the file in case I break something...

	from Peter Biber, 2003
	
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
	pp1 = draw_scan(Q,fig,ax, pt = 0) # set as 2 to hide it so we can see ellipses
	pp2 = draw_scan(P,fig,ax, pt = 1) #pt number assigns color for plotting

	P_corrected = pp2

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

	#debug- draw centers of ellipses from 1st scan
	ax.plot(ctr[:,0], ctr[:,1], 'ko')

	results = []

	for cycle in range(num_cycles):
		# print(" ------------------ cycle: ", cycle, "----------------------" )

		scores = np.zeros(4)
		grad = np.zeros([3,1])
		grad_step_xy = 0.1 #how far we step in x and y when calculating gradients
		grad_step_theta = 0.01
		stepsize = 0.05 #how fast we move in gradient descent

		#compute score for baseline and differential changes in each parameter
		for param in range(4): 

			#get baseline
			x_temp = np.zeros(np.shape(x))
			x_temp[:]  = x[:]

			#differentially change x and y one at a time
			if param == 0:
				x_temp[0] += grad_step_xy
			if param == 1:
				x_temp[1] += grad_step_xy
			if param == 2:
				x_temp[2] += grad_step_theta
			# print("x_temp ",x_temp)

			# 	---------------------------------------------------------
			# 3) For each sample of the second scan: Map the
			# 		reconstructed 2D point into the coordinate frame of
			# 		the first scan according to the parameters.

			#apply transformation
			t_temp = x_temp[:2]
			rot_temp = R(x_temp[2]) 
			P_temp = rot_temp.dot(pp2.T) + t_temp #P_temp is P_corrected with differential changes to each param in x

			#to debug correspondences
			# if (cycle == num_cycles - 1) and (param == 3):
			# 	dc = True
			# else:
			# 	dc = False 
			dc = False
			correspondences = get_correspondence(P_temp,ctr.T,fig,ax, draw = dc)
			# print("c ", correspondences[0], np.shape(correspondences))

			#	----------------------------------------------------------
			#4) Determine the corresponding normal distributions
			# 		for each mapped point.
			score = 0
			for index, i in enumerate(correspondences[0]): 

				mu = E1[int(i)][0][:,None]  	#center of corresponding point cloud from 1st scan (q_i in Biber paper)
				# print("mu", mu, np.shape(mu))
				# print("x_i", P_temp[:,index])
				sigma = E1[int(i)][1]	#associated covariance matrix

				eig = np.linalg.eig(sigma)
				eigenval = eig[0]
				eigenvec = eig[1]

				# Matt's z-score method (not a good idea)-------------------------------------------
				# #get major and minor axis length of corresponding error ellipse
				# #debug - figure out which is which
				# major = np.sqrt(eigenval[1]) + 3 #debug: add minimum size to each ellipse (without this errors explode...)
				# minor = np.sqrt(eigenval[0]) + 3

				# #get rotation of ellipse
				# theta_temp = np.arcsin(eigenvec[0,1]/eigenvec[0,0])

				# #rotate point about origin so that axis of ellipse can be aligned with x,y axis
				# rot_temp = R(theta_temp)
				# pt_rot = rot_temp.dot(pp2[index]) #DEBUG -THIS IS THE PROBLEM??!!!
				# # pt_rot = rot_temp.dot(P[:,index])

				# #figure out how much I need to scale 1STD ellipse to reach point
				# ratio = major/minor			
				# b = np.sqrt( (pt_rot[0]**2)/(ratio**2) + pt_rot[1]**2 )
				# a = ratio*b

				# #use z_score as error
				# z_score = (a / major)**2 #number of standard deviations point is from closest ellipse center

				# #this works(ish) but is not what is actually in the Biber paper
				# score += z_score #keep track of overall scores


				#Biber paper: --------------------------------------------------------------
				# score += -e^( (q.T)(cov)(q) / 2 )
				# 		where: q = distance between x_i and corresponding NDT point  
				# 			   ** assume cov is positive definate

				#reorient each point along the axis of the Normal Distribution ellipse(????) ------------------
				# ang_from_ell_ax = np.arcsin(eigenvec[0,1]/eigenvec[0,0])
				# rot_to_ell_frame = R(ang_from_ell_ax)	
				# #NOTE: x is [x y theta] x' is a point (the notation here is really bad)
				# x_prime = rot_to_ell_frame.dot(P_temp[:,index])[:,None] #Here P represents transformed coord system?
				# # print("before: ", P_temp[:,index], "\n rotated to ell frame: ", x_prime) #makes sense...
				# score += np.exp( (-( x_prime - mu ).T.dot(np.linalg.pinv(sigma)).dot(x_prime - mu) )/2 ) #-----

				# print(P_temp[:,index][:,None])

				#without reorientation
				score += np.exp( (-( P_temp[:,index][:,None] - mu ).T.dot(np.linalg.pinv(sigma)).dot(P_temp[:,index][:,None] - mu) )/2 ) #-----				


			#	-----------------------------------------------------------------
			# 5) The score for the parameters is determined by
			# 		evaluating the distribution for each mapped point
			# 		and summing the result.

			# print(" score: ", score, "param ", param)# "x_temp ", x_temp)
			scores[param] = score
		scores = -scores
		print("scores: ",scores)

		#	-----------------------------------------------------------------
		#6) Calculate a new parameter estimate by trying to
		# 		optimize the score. This is done by performing one
		# 		step of Newton’s Algorithm.

		# print("score: ",scores[-1])
		results = np.append(results, score)

		# Newton's method
		# x1 -> x0 + f(x0)/f'x(0)
		dx = np.zeros(3)
		dx[0] = scores[0] / ((scores[0] - scores[3])/grad_step_xy) 
		dx[1] = scores[1] / ((scores[1] - scores[3])/grad_step_xy) 
		dx[2] = scores[2] / ((scores[2] - scores[3])/ grad_step_theta)
		#NOTE: /0 error here means that changing x results in no score change
		
		print("dx ",dx)

		x += dx[:,None]
		print("x = ",x)

		#DEBUG

		#draw all points progressing through transformation
		rot = R(x[2])
		t = x[0:2]
		P_corrected = rot.dot(pp2.T) + t
		P_corrected = P_corrected.T
		ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),0.025), ls = '', marker = '.', markersize = 20)


	#draw transformed point set
	rot_final = R(x[2])
	t_final = x[:2]

	P_corrected = rot_final.dot(pp2.T) + t_final
	ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.0625), ls = '', marker = '.', markersize = 15)
		
	return x[2], t_final, results
