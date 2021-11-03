import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import misc
from utils import *

#TODO:
#	account for overlapping grid cells

def NDT(Q,P,fig,ax, fid = 10, num_cycles = 1, draw = True, draw_output = True, min_num_pts = 5, along_track_demo = False, output_actual = False, lims = None):

	"""from Peter Biber, 2003
	
		fid -> breaks map down into fid x fid pieces 
		output_actual -> returns transformation ground truth when using along track demo"""

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
	dc = False #draw correspondences

	if along_track_demo == False:
		#get point positions in 2d space and draw 1st and 2nd scans
		pp1 = draw_scan(Q,fig,ax, pt = 2) # set as 2 to hide it so we can see ellipses
		pp2 = draw_scan(P,fig,ax, pt = 1) #pt number assigns color for plotting
	if along_track_demo == True:
		if output_actual == False:
			pp1, pp2 = generate_along_track_data(fig,ax, draw = True, output_actual = False)
		if output_actual == True:
			pp1, pp2, x_actual = generate_along_track_data(fig,ax, draw = True, output_actual = True)

	#for inputting custom point cloud when generating the graphic in the introduction of the paper
	if along_track_demo == 'generate_graphic':
		pp1 = Q.T
		pp2 = P.T
		dc = True
		min_num_pts = 3

	P_corrected = pp2.T

	# Build the NDT of the first scan.
	if type(lims) == np.ndarray: 
		E1 = subdivide_scan(pp1,fig,ax, fidelity = fid, min_num_pts = min_num_pts, pt = 0, lims = lims)
		# print("\n using lims \n", lims)
	else:
		E1 = subdivide_scan(pp1,fig,ax, fidelity = fid, min_num_pts = min_num_pts, pt = 0)
	
	#Initialize the estimate for the parameters 
	x = np.zeros([3,1]) #[ t_x, t_y, theta] }we want to optimize a single euler angle rather than a matrix

	#get 2d coords of centers of each ellipse from E1
	ctr = np.zeros([len(E1),2])
	for idx, c in enumerate(E1):
		ctr[idx,:] = c[0]

	#debug- draw centers of ellipses from 1st scan
	# ax.plot(ctr[:,0], ctr[:,1], 'ko')

	results = []
	x_best = np.zeros([3,1])
	maxscore = 0

	for cycle in range(num_cycles):

		#to debug correspondences
		# if (cycle == num_cycles - 1) and (param == 3):
		# 	dc = True
		# else:
		# 	dc = False 
		if cycle != 0:
			dc = False

		# print("\n P_corrected \n", np.shape(P_corrected))

		xbins_p = np.digitize(P_corrected[0,:], np.linspace(lims[0], lims[1], fid))
		ybins_p = np.digitize(P_corrected[1,:], np.linspace(lims[2], lims[3], fid))
		# print("\n xbins_p \n", xbins_p)
		# print("\n ybins_p \n", ybins_p)

		xbins_ctr = np.digitize(ctr[:,0], np.linspace(lims[2], lims[3], fid))
		ybins_ctr = np.digitize(ctr[:,1], np.linspace(lims[2], lims[3], fid))
		# print("\n xbins_ctr \n", xbins_ctr)
		# print("\n ybins_ctr \n", ybins_ctr)

		correspondences = get_correspondence(P_corrected,ctr.T,fig,ax, draw = dc)
		# print("\n correspondences \n", correspondences[0])

		score = 0
		H = np.zeros([3,3])
		g = np.zeros([3,1])
		matches = 0
		for index, i in enumerate(correspondences[0]): 
			H_i = np.zeros([3,3])
			mu = E1[int(i)][0][:,None]  	#center of corresponding point cloud from 1st scan (q_i in Biber paper)
			sigma = E1[int(i)][1]			#associated covariance matrix
			# print("mu", mu, np.shape(mu))
			# print("x_i", P_corrected[:,index])
			eig = np.linalg.eig(sigma)
			eigenval = eig[0] #not needed?
			eigenvec = eig[1] #not needed?

			q = (P_corrected[:,index][:,None] - mu) #q is the distance between a point and the nearest ellipse center

			E = np.linalg.pinv(sigma)

			## was this: Nearest-Neighbor correspondance ------------------------------
			## score_i = np.exp( (-(q.T).dot(E).dot(q) ) /2 ) # according to Biber
			## score_i = np.exp( (-(q.T).dot(E).dot(q) ) /40 ) #more forgiving -> voxel size is around 40 in this case. Converges in ~70
			## score_i = (q).T.dot(E).dot(q) #Matt's method -> WORKS (slowly). Converges in ~400
			# score_i = np.exp( (-(q.T).dot(E).dot(q) ) /200 ) #trying to get best performance for paper...
			# score += score_i
			## ------------------------------------------------------------------------

			# NEW 11/3 - voxel-based correspondance ----------------------------------
			# trying this to use more robust correspondance metrics  
			#exclude points from score if they do not fall in same voxel as corresponding distribution
				#slightly less efficient for first iterations but should achieve same final accuracy... just looking for results here...
			# print(np.shape(xbins_p))

			# if the point in consideration is within the same x and y bins as the corresponding distribution center
			#BUG IS HERE??
			if (xbins_p[index] == xbins_ctr[i.astype(int)]) and ( ybins_p[index] == ybins_ctr[i.astype(int)]):
				score_i = np.exp( (-(q.T).dot(E).dot(q) ) /200 ) # according to Biber
				score += score_i
				matches += 1 #for debug
				# print(xbins_p[index], xbins_ctr[i.astype(int)])
			else:
				score_i = 0
			#--------------------------------------------------------------------------


			#DEBUG THIS - be careful of mixing up x and p
			#get jacobian
			#using func from utils that I wrote for ICP
			J = jacobian(x, P_corrected[:,index])  
			#as described in Biber paper
			# J2 = np.array([[1, 0, (-x[0]*np.sin(x[2]) - x[1]*np.cos(x[2]))[0] ],
						  # [0, 1, (x[0]*np.cos(x[2]) - x[1]*np.sin(x[2]))[0]  ]]) 
			# print(J)
			# print(J2)

			#update gradient-----
			#J == dq/dp_i

			#this
			g += (q).T.dot(E).dot(J).T*(-score_i)
			# does the same thing as:
			# g_i = np.zeros([3,1])
			# for ct in range(3):
			# 	g_i[ct] = (q).T.dot(E).dot(J[:,ct][:,None])*(-score_i)
			# g += g_i
		
			#update Hessian----- H(f(x)) == J(grad_f(x))

			# index h_i and h_j are used in place of i and j since I already used those variables...
			for h_i in range(np.shape(H)[0]):
				for h_j in range(np.shape(H)[1]):

					#manually calculate 2nd deriv
					if h_i == 2 and h_j == 2:
						d2q_dxidxj = np.array([-x[0]*np.cos(x[2]) + x[1]*np.sin(x[2]), -x[0]*np.sin(x[2]) - x[1]*np.cos(x[2])])
						# d2q_dxidxj = np.array([-P_corrected[0,index]*np.cos(x[2]) + P_corrected[1,index]*np.sin(x[2]), -P_corrected[0,index]*np.sin(x[2]) - P_corrected[1,index]*np.cos(x[2])])
					else:
						d2q_dxidxj = np.zeros([2,1])

					# print("\n J[:,h_i][:,None] \n", J[:,h_i][:,None])
					# print("\n J[:,h_j][:,None] \n", J[:,h_j][:,None])

					# #calculate component of ith summand of H
					H_i[h_i,h_j] = score_i*( (-q.T.dot(E).dot(J[:,h_i][:,None])).dot( -q.T.dot(E).dot(J[:,h_j][:,None]) )
											+ (-q.T.dot(E).dot(d2q_dxidxj)) + (-J[:,h_j].T.dot(E).dot(J[:,h_i]))) 

					# H[h_i,h_j] += H_i[h_i,h_j]
			H += H_i

		# print(score)
		results = np.append(results, score)

		#	-----------------------------------------------------------------
		#6) Calculate a new parameter estimate by trying to
		# 		optimize the score. This is done by performing one
		# 		step of Newton’s Algorithm.
		#	(H)(delP) = -g
		#		H = hessian
		#		delP = how far we need to change parameters
		#		g = transposed gradient of f

		#THIS IS CURRENTLY WRONG:
		# ((H.T)(W)(H) + I*10e-6(max(eig)))^-1   <- correct way to do this WITHIN main inverse NOT AFTER
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#check if H is positive definite (fixes non-invertability issue with HTWH)
		posdef = np.all(np.linalg.eigvals(np.linalg.pinv(H)) > 0)
		# if posdef == False:
		# 	print("WARNING: not posdef")
		lam = 10e-6
		while posdef == False:
			H = H + lam*np.identity(3)
			posdef = np.all(np.linalg.eigvals(np.linalg.pinv(H)) > 0)
			# print(posdef)
			lam = lam*2
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


		# print("H^-1", np.linalg.pinv(H))

		if score > maxscore:
			maxscore = score
			x_best[:] = x[:]
			# print("maxscore: \n", maxscore, "\n x_best: \n", x_best)
		# else:
		# 	x[:] = x_best[:] #prevents resetting to zero

		#was this
		dx = np.linalg.pinv(H).dot(-g)
		#testing
		# dx = -g.dot(np.linalg.pinv(H))

		# x += dx #Biber
		x -= dx #Matt
		# dx = (-g.T).dot(np.linalg.pinv(H))
		# if cycle%10 == 0:
		# 	print(" dx = \n",dx)
		# 	print("g", g)
		# 	print("H", H)
		# x -= dx.T

		# dx_dumb = np.linalg.pinv(H.dot(np.linalg.pinv(-g.T)))
		# x += dx_dumb.T
		# print("x = ",x)

		#draw all points progressing through transformation
		rot = R(x[2])
		t = x[0:2]
		P_corrected = rot.dot(pp2.T) + t #was this

		#plot progression
		# ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),0.0025), ls = '', marker = '.', markersize = 20)

	#draw transformed point set (last point)
	# rot_final = R(x[2])
	# t_final = x[:2]
	#best overall
	rot_final = R(x_best[2])
	t_final = x_best[:2]


	P_corrected = rot_final.dot(pp2.T) + t_final
	if draw_output:
		ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.0625), ls = '', marker = '.', markersize = 15)

	#draw correspondences of final state
	# get_correspondence(P_corrected,ctr.T,fig,ax, draw = True)

	# print(matches)

	#TODO- DEBUG THIS
	if (output_actual == False):
		return x_best[2], t_final, results

	if (output_actual == True) and (along_track_demo == True):
		return x_best[2], t_final, results, x_actual




# def NDT_old(Q,P,fig,ax, fid = 10, num_cycles = 1, draw = True):

# 	"""
# 	keeping this in the file in case I break something...

# 	from Peter Biber, 2003
	
# 		fid -> breaks map down into fid x fid pieces """

# 	# 1) Build the NDT of the first scan.
# 	# 2) Initialize the estimate for the parameters (by zero or
# 	# 		by using odometry data).
# 	# 3) For each sample of the second scan: Map the
# 	# 		reconstructed 2D point into the coordinate frame of
# 	# 		the first scan according to the parameters.
# 	# 4) Determine the corresponding normal distributions
# 	# 		for each mapped point.
# 	# 5) The score for the parameters is determined by
# 	# 		evaluating the distribution for each mapped point
# 	# 		and summing the result.
# 	# 6) Calculate a new parameter estimate by trying to
# 	# 		optimize the score. This is done by performing one
# 	# 		step of Newton’s Algorithm.
# 	# 7) Goto 3 until a convergence criterion is met.

# 	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 	# speed things up by decimating points
# 	# print("before ", np.shape(P))
# 	# P = P[::10]
# 	# print("after ", np.shape(P))

# 	#get point positions in 2d space and draw 1st and 2nd scans
# 	pp1 = draw_scan(Q,fig,ax, pt = 0) # set as 2 to hide it so we can see ellipses
# 	pp2 = draw_scan(P,fig,ax, pt = 1) #pt number assigns color for plotting

# 	P_corrected = pp2

# 	#	---------------------------------------------------------
# 	# 1) Build the NDT of the first scan.
# 	E1 = subdivide_scan(pp1,fig,ax, fidelity = fid, pt =0)

# 	#	----------------------------------------------------------
# 	# 2) Initialize the estimate for the parameters (by zero or
# 	# 		by using odometry data)
# 	x = np.zeros([3,1]) #[ t_x, t_y, theta] }we want to optimize a single euler angle rather than a matrix

# 	#get 2d coords of centers of each ellipse from E1
# 	ctr = np.zeros([len(E1),2])
# 	for idx, c in enumerate(E1):
# 		ctr[idx,:] = c[0]

# 	#debug- draw centers of ellipses from 1st scan
# 	ax.plot(ctr[:,0], ctr[:,1], 'ko')

# 	results = []

# 	for cycle in range(num_cycles):
# 		# print(" ------------------ cycle: ", cycle, "----------------------" )

# 		scores = np.zeros(4)
# 		grad = np.zeros([3,1])
# 		grad_step_xy = 0.1 #how far we step in x and y when calculating gradients
# 		grad_step_theta = 0.01
# 		stepsize = 0.05 #how fast we move in gradient descent

# 		#compute score for baseline and differential changes in each parameter
# 		for param in range(4): 

# 			#get baseline
# 			x_temp = np.zeros(np.shape(x))
# 			x_temp[:]  = x[:]

# 			#differentially change x and y one at a time
# 			if param == 0:
# 				x_temp[0] += grad_step_xy
# 			if param == 1:
# 				x_temp[1] += grad_step_xy
# 			if param == 2:
# 				x_temp[2] += grad_step_theta
# 			# print("x_temp ",x_temp)

# 			# 	---------------------------------------------------------
# 			# 3) For each sample of the second scan: Map the
# 			# 		reconstructed 2D point into the coordinate frame of
# 			# 		the first scan according to the parameters.

# 			#apply transformation
# 			t_temp = x_temp[:2]
# 			rot_temp = R(x_temp[2]) 
# 			P_temp = rot_temp.dot(pp2.T) + t_temp #P_temp is P_corrected with differential changes to each param in x

# 			#to debug correspondences
# 			# if (cycle == num_cycles - 1) and (param == 3):
# 			# 	dc = True
# 			# else:
# 			# 	dc = False 
# 			dc = False
# 			correspondences = get_correspondence(P_temp,ctr.T,fig,ax, draw = dc)
# 			# print("c ", correspondences[0], np.shape(correspondences))

# 			#	----------------------------------------------------------
# 			#4) Determine the corresponding normal distributions
# 			# 		for each mapped point.
# 			score = 0
# 			for index, i in enumerate(correspondences[0]): 

# 				mu = E1[int(i)][0][:,None]  	#center of corresponding point cloud from 1st scan (q_i in Biber paper)
# 				# print("mu", mu, np.shape(mu))
# 				# print("x_i", P_temp[:,index])
# 				sigma = E1[int(i)][1]	#associated covariance matrix

# 				eig = np.linalg.eig(sigma)
# 				eigenval = eig[0]
# 				eigenvec = eig[1]

# 				# Matt's z-score method (not a good idea)-------------------------------------------
# 				# #get major and minor axis length of corresponding error ellipse
# 				# #debug - figure out which is which
# 				major = np.sqrt(eigenval[1]) + 3 #debug: add minimum size to each ellipse (without this errors explode...)
# 				minor = np.sqrt(eigenval[0]) + 3

# 				#get rotation of ellipse
# 				theta_temp = np.arcsin(eigenvec[0,1]/eigenvec[0,0])

# 				#rotate point about origin so that axis of ellipse can be aligned with x,y axis
# 				rot_temp = R(theta_temp)
# 				pt_rot = rot_temp.dot(pp2[index]) #DEBUG -THIS IS THE PROBLEM??!!!
# 				# pt_rot = rot_temp.dot(P[:,index])

# 				#figure out how much I need to scale 1STD ellipse to reach point
# 				ratio = major/minor			
# 				b = np.sqrt( (pt_rot[0]**2)/(ratio**2) + pt_rot[1]**2 )
# 				a = ratio*b

# 				#use z_score as error
# 				z_score = (a / major)#**2 #number of standard deviations point is from closest ellipse center

# 				#this works(ish) but is not what is actually in the Biber paper
# 				score += z_score #keep track of overall scores


# 				#Biber paper: --------------------------------------------------------------
# 				# score += -e^( (q.T)(cov)(q) / 2 )
# 				# 		where: q = distance between x_i and corresponding NDT point  
# 				# 			   ** assume cov is positive definate

# 				#reorient each point along the axis of the Normal Distribution ellipse(????) ------------------
# 				# ang_from_ell_ax = np.arcsin(eigenvec[0,1]/eigenvec[0,0])
# 				# rot_to_ell_frame = R(ang_from_ell_ax)	
# 				# #NOTE: x is [x y theta] x' is a point (the notation here is really bad)
# 				# x_prime = rot_to_ell_frame.dot(P_temp[:,index])[:,None] #Here P represents transformed coord system?
# 				# # print("before: ", P_temp[:,index], "\n rotated to ell frame: ", x_prime) #makes sense...
# 				# score += np.exp( (-( x_prime - mu ).T.dot(np.linalg.pinv(sigma)).dot(x_prime - mu) )/2 ) #-----

# 				# print(P_temp[:,index][:,None])

# 				#without reorientation
# 				# score += np.exp( (-( P_temp[:,index][:,None] - mu ).T.dot(np.linalg.pinv(sigma)).dot(P_temp[:,index][:,None] - mu) )/2 ) #-----				


# 			#	-----------------------------------------------------------------
# 			# 5) The score for the parameters is determined by
# 			# 		evaluating the distribution for each mapped point
# 			# 		and summing the result.

# 			# print(" score: ", score, "param ", param)# "x_temp ", x_temp)
# 			scores[param] = score
# 		scores = -scores
# 		print("scores: ",scores)

# 		#	-----------------------------------------------------------------
# 		#6) Calculate a new parameter estimate by trying to
# 		# 		optimize the score. This is done by performing one
# 		# 		step of Newton’s Algorithm.

# 		# print("score: ",scores[-1])
# 		results = np.append(results, score)

# 		# Newton's method
# 		# x1 -> x0 + f(x0)/f'x(0)
# 		dx = np.zeros(3)
# 		dx[0] = scores[0] / ((scores[0] - scores[3])/grad_step_xy) 
# 		dx[1] = scores[1] / ((scores[1] - scores[3])/grad_step_xy) 
# 		dx[2] = scores[2] / ((scores[2] - scores[3])/ grad_step_theta)
# 		#NOTE: /0 error here means that changing x results in no score change
		
# 		print("dx ",dx)

# 		x += stepsize*dx[:,None]
# 		print("x = ",x)

# 		#DEBUG

# 		#draw all points progressing through transformation
# 		rot = R(x[2])
# 		t = x[0:2]
# 		P_corrected = rot.dot(pp2.T) + t
# 		P_corrected = P_corrected.T
# 		ax.plot(P_corrected.T[0,:], P_corrected.T[1,:], color = (1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),1-(cycle+1)/(num_cycles+1),0.025), ls = '', marker = '.', markersize = 20)


# 	#draw transformed point set
# 	rot_final = R(x[2])
# 	t_final = x[:2]

# 	P_corrected = rot_final.dot(pp2.T) + t_final
# 	ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.0625), ls = '', marker = '.', markersize = 15)
		
# 	return x[2], t_final, results
