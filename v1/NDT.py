import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

def NDT(Q,P,fig,ax, fid = 10, num_cycles = 1, draw = True):

	"""from Peter Biber, 2003
	
		fid....breaks map down into fid x fid pieces """

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

	#get point positions in 2d space and draw 1st and 2nd scans
	pp1 = draw_scan(Q,fig,ax, pt = 2) # set as 2 to hide it so we can see ellipses
	pp2 = draw_scan(P,fig,ax, pt = 1) #pt number assigns color for plotting

	#subdivide 
	E1 = subdivide_scan(pp1,fig,ax, fidelity = fid, pt =0)

	#get 2d coords of centers of each ellipse from E1
	ctr = np.zeros([len(E1),2])
	for idx, c in enumerate(E1):
		ctr[idx,:] = c[0]

	#for debug: LSICP between 2nd scan and center points of first ellipses
	# P_corrected = ICP_least_squares(pp2.T, ctr.T, fig, ax, num_cycles = num_cycles, draw = True)

	#for debug: downsample number of points in scan2 to make this run faster (mpl is slow)

	for cycle in range(num_cycles):
		# get correspondences between 2nd scan points and 1st scan ellipses
		correspondences = get_correspondence(pp2.T,ctr.T,fig,ax)
		# print("c ", correspondences, np.shape(correspondences))

		# loop through correspondences and calculate the z score of each point 
		score = 0
		for index, i in enumerate(correspondences[0]): 

			mu = E1[int(i)][0]
			sigma = E1[int(i)][1]

			eig = np.linalg.eig(sigma)
			eigenval = eig[0]
			eigenvec = eig[1]

			#get major and minor axis length of corresponding error ellipse
			major = np.sqrt(eigenval[0])
			minor = np.sqrt(eigenval[1])

			#get rotation of ellipse
			theta = -np.arcsin(eigenvec[0,1]/eigenvec[0,0])

			#rotate point about origin so that axis of ellipse can be aligned with x,y axis
			rot = R(theta)
			pt_rot = rot.dot(pp2[index])

			#figure out how much I need to scale 1STD ellipse to reach point
			ratio = major/minor			
			b = np.sqrt( (pt_rot[0]**2)/(ratio**2) + pt_rot[1]**2 )
			a = ratio*b

			z_score = a / major
			score += z_score

		print("score: ", score)

		# update R and t to lower score using Newton's method  
		r = None
		t = None

	return r, t, E1

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


	#TODO- 
	#	check out point to plane metric- usually gives better results with imperfect points
	#		assumes poitns are sampled from real world surface
	#		uses least squares approach -> gaussian error metric?
	#	outliar rejection techniques	
	#	RN THIS ONLY WORKS IF P and Q ARE THE SAME LENGTH
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

		#init Hessian
		H = np.zeros([3,3])
		g = np.zeros([3,1])
		chi = 0

		if draw == True and cycle == num_cycles-1:
			draw_this_time = True
		else:
			draw_this_time = False

		correspondences = get_correspondence(P_corrected, true_data, fig, ax, draw = draw_this_time)
		# print(correspondences)

		#get H, g, chi
		for i in range(np.shape(correspondences)[1]):
			p =  P_corrected[:,i] #was this
			# p = P[:,i]	#debug: not this??
			q = true_data[:,int(correspondences[0,i])][:,None]
			# print(q)
			# print(int(correspondences[0,i]))

			# print("p ", p, np.shape(p))
			# print("q ", q, np.shape(q))

			err = error(x,p,q)
			# print("error2 ", err, np.shape(err))
			weight = 1 #TODO: replace with lambda func at some point...

			J = jacobian(x, p)

			H += weight * J.T.dot(J)
			# print(np.shape(J.T))
			g += weight * J.T.dot(err)
			chi += err.T * err

		# print(H, g, chi)

		dx = np.linalg.lstsq(H, -g, rcond=None)[0] #TODO: recreate this func
		x += dx
		rot = R(x[2]) #.T #aha! -> this should not be transposed lol
		# print(x[2])
		# print("rot ", rot)
		t = x[0:2]
		# print("ang before normalize ", x[2])
		x[2] = np.arctan2(np.sin(x[2]), np.cos(x[2])) # normalize angle
		# print("ang after normalize ", x[2])


		P_corrected = rot.dot(P_corrected) + t
		P_corrected = np.squeeze(P_corrected)
		# print("P_corrected ",np.shape(P_corrected))

	ax.plot(P_corrected[0,:], P_corrected[1,:], color = (1,0,0,0.0625), ls = '', marker = '.', markersize = 20)

	print(x)

	return P_corrected


def R(theta):
	"""Rotation Matrix"""
	mat = np.array([[np.cos(theta), -np.sin(theta)],
					[np.sin(theta),  np.cos(theta)]])
	mat = np.squeeze(mat)
	# print("mat ", np.shape(mat))
	return mat 


def dR(theta):
	"""derivative of rotation matrix"""
	return np.array([[-np.sin(theta), -np.cos(theta)],
					[np.cos(theta),  -np.sin(theta)]])

def jacobian(x, p_point):
	"""outputs: (2,3) np array"""
	theta = x[2]
	J = np.zeros((2, 3))
	J[0:2, 0:2] = np.identity(2)
	J[0:2, [2]] = dR(0).dot(p_point)[:,None] #need to increase dims of p_point
	# print("jacobian: ", J, np.shape(J))
	return J

def error(x, p_point, q_point):
	"""outputs: (2,1) np array"""

	# ~bug hunt~ I think the issue is in here somewhere?

	# print("x ",x)
	rotation = R(x[2]) #why is this always identity???
	rotation = np.squeeze(rotation)
	# print("after squeeze ", rotation, np.shape(rotation))
	translation = x[0:2]
	# prediction = rotation.T.dot(p_point).T  + translation #was this mess

	prediction = rotation.dot(p_point)  + translation.T #trying this
	# print("prediction ", prediction, np.shape(prediction))

	err = prediction.T - q_point

	# print("error ", err, np.shape(err))

	return err

def get_cross_cov(P,Q,correspondence):

	#TODO - get rid of outliars (points with weight << 0.01)
	weight = 1 #can be adjusted dynamically

	cov = np.zeros([2,2])

	for i in range(np.shape(correspondence)[1]):
		closest = Q[:,int(correspondence[0,i])]
		# print("P ", P[:,i].T)
		# print("Q ", closest)
		a =  weight * closest[:,None].dot(P[:,i][:,None].T) #had to do some witchcraft to transpose 1D arrays [:,None] adds another axis
		# print("a ", a)
		cov += a

	# print("cov = ", cov)

	return cov


def get_correspondence(P,Q, fig, ax, draw = False):

	"""generates an array containing a the closest point on Q for each point of P as well as the minimum distnace for each"""

	#init array to store closest points on Q for each point on the moved data
	correspondences = np.ones([2,np.shape(P)[1]])
	correspondences[1,:] = 1e10 #set closest distance arbitrarily far away

	for p in range(np.shape(P)[1]):

		for q in range(np.shape(Q)[1]):
			#calculate distance squared between points
			d = (P[0,p] - Q[0,q])**2 + (P[1,p] - Q[1,q])**2
			if d < correspondences[1,p]:
				correspondences[0,p] = q #save nearest neighbor
				correspondences[1,p] = d #save distance from neighbor

		if draw == True:
			#plot lines between nearest neighbors
			# print("drawing line between")
			# print(P[:,p])
			# print(Q[:,int(closest[0,p])])
			# print("---")
			start = P[:,p]
			stop = Q[:,int(correspondences[0,p])]
			ax.plot([start[0], stop[0]], [start[1], stop[1]],'g')

	return correspondences

def subdivide_scan(pp, fig, ax, fidelity = 5, overlap = False, nstd = 2, pt = 0):

	#color differently depending on which scan
	if pt == 0:
		color = 'green'
	if pt == 1:
		color = 'blue'

	E = []

	#define bounding box surrounding all points
	minx = np.min(pp[:,0])
	maxx = np.max(pp[:,0])
	miny = np.min(pp[:,1])
	maxy = np.max(pp[:,1])

	X = np.linspace(minx,maxx,fidelity)
	Y = np.linspace(miny,maxy,fidelity)

	for x in range(len(X)-1):
		for y in range(len(Y)-1):

			#narrow down the points to be within the y range
			within_y = pp[pp[:,1] > Y[y]]
			within_y = within_y[within_y[:,1] < Y[y+1] ]

			#narrow down the points to be within x range
			within_x = within_y[within_y[:,0] > X[x]]
			within_box = within_x[within_x[:,0] < X[x+1]]  

			# print(np.shape(within_box))
			# print(within_box)

			if np.shape(within_box)[0] > 2:

				mu, sigma = fit_gaussian(within_box)

				eig = np.linalg.eig(sigma)
				eigenval = eig[0]
				eigenvec = eig[1]

				rot = -np.rad2deg(np.arcsin(eigenvec[0,1]/eigenvec[0,0]))
				width = 2*nstd*np.sqrt(eigenval[0])
				height = 2*nstd*np.sqrt(eigenval[1])

				ell = Ellipse((mu[0],mu[1]),width, height, angle = rot, fill = False, color = color)
				ax.add_patch(ell)

				E.append((mu, sigma))

	return E

def draw_scan(scan, fig, ax, FOV = 90, pt = 0):

	#init array to store xy pos of points from lidar relative to frame of robot
	point_pos = np.zeros([np.shape(scan)[0],2])

	if pt == 0:
		color = 'g.'
	if pt == 1:
		color = 'b.'
	if pt ==2:
		color = (0,0,0,0)

	#draw base point
	ax.plot(0,0,'rx', markersize = 5)

	#draw FOV boundary
	# ax.plot([0,100],[0,100],'r--', lw = 1)
	# ax.plot([0,-200],[0,200],'r--', lw = 1)

	for i in range(np.shape(scan)[0]):

		step = i*(FOV/np.shape(scan)[0]) - FOV/2

		x = np.sin(np.deg2rad(step))*scan[i]
		y = np.cos(np.deg2rad(step))*scan[i]

		ax.plot(x,y, color)

		#store estimated xy of each point relative to robot's current position
		point_pos[i,0] = x
		point_pos[i,1] = y

	plt.draw()

	return point_pos

def fit_gaussian(points):

	"""inputs: 2D np array of points in shape (n, 2)
		outputs: distribution with center and standard deviation"""

	ndims = len(np.shape(points))

	if ndims == 1:
		mean = np.mean(points)
		std = np.sqrt(np.sum( (points - mean)**2 ) / np.shape(points)[0] )
		output = (mean, std)

	if ndims == 2:

		mu_x = np.mean(points[:,0])
		mu_y = np.mean(points[:,1])
		mu = np.array([mu_x, mu_y])

		#  | stdx^2   	   rho*stdx*stdy|
		#  | rho*stdx*stdy	     stdy^2 |

		#	ellipse of 1 standard deviations:
		#		major and minor axis are sqrt of eigenvalues of this matrix
		#		matrix is always symetric -> eigenvectors always orthagonal

		#covariance of x and y
		# cov_xy = np.cov(points.T) #not needed?

		#standard deviations
		std_x = np.sqrt(np.sum( (points[:,0] - mu_x)**2 ) / np.shape(points)[0] )
		std_y = np.sqrt(np.sum( (points[:,1] - mu_y)**2 ) / np.shape(points)[0] )

		#expected value
		E = np.mean( (points[:,0] - mu_x) * (points[:,1] - mu_y) )

		# 	rho: pearson correlation coefficient
		rho = E / (std_x*std_y)

		sigma = np.array([[std_x**2,  rho*std_x*std_y],
						  [rho*std_x*std_y, std_y**2 ]])


		output = (mu, sigma)
	

	return output

def draw_ellipse():

	'''draw standard deviation ellipse on subset of points'''

	pass

if __name__ == "__main__":


	fig = plt.figure()
	ax = fig.add_subplot()

	# ax.set_xlim(-200,200)
	# ax.set_ylim(0,400)
	ax.set_aspect('equal')

	dataset = np.load("data/train_dxdy_100k.npy")

	dat = dataset[:,:50]

	rand = int(np.random.rand()*np.shape(dat)[0])

	draw_scan(dat[rand],fig,ax)

	plt.pause(10)