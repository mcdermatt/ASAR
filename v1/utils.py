import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy import misc


def R(theta):
	"""Rotation Matrix"""
	mat = np.array([[np.cos(theta), -np.sin(theta)],
					[np.sin(theta),  np.cos(theta)]])
	mat = np.squeeze(mat)
	# print("mat ", np.shape(mat))
	return mat 

def R_alt(r):
	"""get rotatation matrix of secondary axis from first rotation matrix"""

	# # was this
	# if r[0,0] < 0:
	# 	arr = np.array([[np.cos(np.arccos(r[0,0]) + np.pi/2), np.sin(np.arcsin(-r[0,1]) + np.pi/2)],
	# 					 [np.sin(np.arcsin(r[1,0]) + np.pi/2), np.cos(np.arccos(r[1,1]) + np.pi/2)]])
	# else:
	# 	arr = np.array([[np.cos(np.arccos(r[0,0]) + np.pi/2), -np.sin(np.arcsin(-r[0,1]) + np.pi/2)],
	# 				 [np.sin(np.arcsin(r[1,0]) + np.pi/2), np.cos(np.arccos(r[1,1]) + np.pi/2)]])

	#fixes issues with arcsin/ arccos always between +/- (pi/2)
	theta = np.arctan2(r[1,0],r[0,0])
	theta += np.pi/2
	arr = R(theta)

	return arr

def dR(theta):
	"""derivative of rotation matrix"""
	return np.array([[-np.sin(theta), -np.cos(theta)],
					[np.cos(theta),  -np.sin(theta)]])

def jacobian(x, p_point):
	"""outputs: (2,3) np array"""
	theta = x[2]
	J = np.zeros((2, 3))
	J[0:2, 0:2] = np.identity(2)
	# J[0:2, [2]] = dR(0).dot(p_point)[:,None] #was this -> wrong!!!
	J[0:2, [2]] = dR(theta[0]).dot(p_point)[:,None] #changed to this 10/5

	return J

def error(x, p_point, q_point):
	"""
	calculates error to be used in Least Squares ICP
	outputs: (2,1) np array"""

	rotation = R(x[2])
	rotation = np.squeeze(rotation)
	translation = x[0:2]

	prediction = rotation.dot(p_point)  + translation.T
	err = prediction.T - q_point

	return err

def get_cross_cov(P,Q,correspondence):

	#TODO - get rid of outliars (points with weight << 0.01)
	weight = 1 #can be adjusted dynamically

	cov = np.zeros([2,2])

	for i in range(np.shape(correspondence)[1]):
		closest = Q[:,int(correspondence[0,i])]
		a =  weight * closest[:,None].dot(P[:,i][:,None].T) #had to do some witchcraft to transpose 1D arrays [:,None] adds another axis
		cov += a
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
			start = P[:,p]
			stop = Q[:,int(correspondences[0,p])]
			ax.plot([start[0], stop[0]], [start[1], stop[1]],'k--', lw = 1)

	return correspondences

def subdivide_scan(pp, fig, ax, fidelity = 5, overlap = False, 
	min_num_pts = 5, nstd = 2, pt = 0, flag = False, output_lims = False, lims = None):
	#flag is for making demo figure where extended feature takes up 3 voxels in x direction

	#color differently depending on which scan
	if pt == 0:
		color = 'green'
	if pt == 1:
		color = 'blue'
	#set pt to 2 to not draw

	E = []

	#use existing voxel positions established at beginning of scan-matching routine
	if type(lims) == np.ndarray:
		minx = lims[0]
		maxx = lims[1]
		miny = lims[2]
		maxy = lims[3]
	else:
		#define bounding box surrounding all points
		#adaptive grid cell size
		# minx = np.min(pp[:,0])
		# maxx = np.max(pp[:,0])
		# miny = np.min(pp[:,1])
		# maxy = np.max(pp[:,1])

		bound = 250
		minx = np.min(-bound)
		maxx = np.max(bound)
		miny = np.min(-bound)
		maxy = np.max(bound)

		# minx = -1
		# maxx = 31
		# miny = -16
		# maxy = 16


		#draw grid lines on figure
		xticks = np.linspace(minx,maxx,fidelity)
		ax.axes.xaxis.set_ticks(xticks)
		yticks = np.linspace(miny,maxy,fidelity)
		ax.axes.yaxis.set_ticks(yticks)
		ax.grid(color=(0,0,0), linestyle='-', linewidth=0.5)
		ax.set_xlim([minx,maxx])
		ax.set_ylim([miny,maxy])

	# DEBUG: fixed grid cell size - need to change with each scan!!!
	# minx = np.min(-500)
	# maxx = np.max(500)
	# miny = np.min(-500)
	# maxy = np.max(500)

	X = np.linspace(minx,maxx,fidelity)
	Y = np.linspace(miny,maxy,fidelity)

	if flag:
		X = np.linspace(-400,400,4)

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

			if np.shape(within_box)[0] > min_num_pts-1:

				mu, sigma = fit_gaussian(within_box)

				eig = np.linalg.eig(sigma)
				eigenval = eig[0]
				eigenvec = eig[1]

				rot = -np.rad2deg(np.arctan(eigenvec[0,1]/eigenvec[0,0]))
				#was this
				# width = 2*nstd*np.sqrt(eigenval[0])
				# height = 2*nstd*np.sqrt(eigenval[1])
				#debug
				width = 2*nstd*np.sqrt(abs(eigenval[0]))
				height = 2*nstd*np.sqrt(abs(eigenval[1]))


				if pt != 2:
					ell = Ellipse((mu[0],mu[1]),width, height, angle = rot, fill = False, color = color)
					ax.add_patch(ell)

				E.append((mu, sigma,  np.shape(within_box)[0]))

	if output_lims == False:
		return E
	else:
		lims = np.array([minx, maxx, miny, maxy])
		return E, lims

def draw_scan(scan, fig, ax, FOV = 60, pt = 0, hitters = None, ignore_boundary = False):

	#init array to store xy pos of points from lidar relative to frame of robot
	point_pos = np.zeros([np.shape(scan)[0],2])

	if pt == 0:
		color = 'g.'
	if pt == 1:
		color = 'b.'
	if pt ==2:
		color = (0,0,0,0)

	#draw base point
	if ignore_boundary == False:
		ax.plot(0,0,'rx', markersize = 5)
	# if ignore_boundary == True:
		# ax.plot(0,0,'g.', markersize  = 20) #not sure if I should plot this...

	#draw FOV boundary
	# ax.plot([0,100],[0,100],'r--', lw = 1)
	# ax.plot([0,-200],[0,200],'r--', lw = 1)

	for i in range(np.shape(scan)[0]):

		step = i*(FOV/np.shape(scan)[0]) - FOV/2

		x = np.sin(np.deg2rad(step))*scan[i]
		y = np.cos(np.deg2rad(step))*scan[i]

		#normal
		if ignore_boundary == False:
			if pt != 2: #runs faster without plotting clear pts
				ax.plot(x,y, color)
		if ignore_boundary == True:
			if (pt != 2) and (hitters[i-1] == 1):
				ax.plot(x,y,'r.', markersize = 2) 

		#store estimated xy of each point relative to robot's current position
		point_pos[i,0] = x
		point_pos[i,1] = y

	plt.draw()

	return point_pos

# def generate_along_track_data(fig,ax,draw = True, output_actual = False):

# 	""" Generates straight road scan (ambiguous case) """

# 	npts = 4200 #4200
# 	tscale = 10
# 	noise_scale = 2

# 	pp1 = np.zeros([npts,2])
# 	pp2 = np.zeros([npts,2])

# 	#ADJUST NET ROTATION HERE
# 	# theta = np.random.randn()*0.1 #+np.pi
# 	# t = np.random.randn(2)*tscale
# 	theta = 0.1
# 	t = np.array([5,10])

# 	rot = R(theta)

# 	#moves half of points to left wall and half to right
# 	xshift = np.ones(npts)*-125
# 	xshift[(npts//2):] = 125
# 	yshift = np.zeros(npts) - 250
# 	yshift[(npts//2):] = -750

# 	for i in range(npts):
# 		pp1[i,0] = xshift[i] #+ np.random.randn()*x_noise_scale
# 		pp1[i,1] = i*1000/npts + yshift[i] #+ np.random.randn()*y_noise_scale

# 	# pp1 = R(np.pi/3).dot(pp1.T)
# 	# pp1 = pp1.T

# 	pp2[:,:] = pp1[:,:]
# 	pp1 += np.random.randn(npts,2)*noise_scale
# 	pp2 += np.random.randn(npts,2)*noise_scale

# 	#transform scan2
# 	pp2 += t
# 	pp2 = rot.dot(pp2.T)
# 	pp2 = pp2.T

# 	# ax.plot(pp1[:,0], pp1[:,1], color = (0.25,0.8,0.25,0.0375), ls = '', marker = '.', markersize = 20)
# 	# ax.plot(pp2[:,0], pp2[:,1], color = (0.25,0.25,0.8,0.0375), ls = '', marker = '.', markersize = 20)

# 	ax.plot(pp1[:,0], pp1[:,1], color = (0.25,0.8,0.25), ls = '', marker = '.', markersize = 1)
# 	ax.plot(pp2[:,0], pp2[:,1], color = (0.25,0.25,0.8), ls = '', marker = '.', markersize = 1)

# 	if output_actual == False:
# 		return pp1, pp2
# 	if output_actual == True:
# 		x_actual = np.array([t[0], t[1], theta])
# 		return pp1, pp2, x_actual

def generate_along_track_data(fig,ax,draw = True, output_actual = False):

	""" Generates T shaped scan (non-ambiguous case) """

	npts =  4200 #16800
	tscale = 5#10
	noise_scale = 2

	pp1 = np.zeros([npts,2])
	pp2 = np.zeros([npts,2])


	#ADJUST NET ROTATION HERE
	theta = np.random.randn()*0.1 #0.1
	t = np.random.randn(2)*tscale
	# theta = 0.1
	# t = np.array([5,10])

	rot = R(theta)


	#top part of T
	pp1[:int(npts/3), :] = np.array([np.linspace(-250,250,int(npts/3)),125+10*np.sin(np.linspace(-2,2,int(npts/3)))]).T #wavy
	# pp1[:int(npts/3), :] = np.array([np.linspace(-250,250,int(npts/3)),125*np.ones(int(npts/3))]).T #flat
	#left horizontal
	pp1[int(npts/3):int(npts/2), :] = np.array([np.linspace(-250,-75,int(npts/6)),np.ones(int(npts/6))*-75]).T
	#left vertical
	pp1[int(npts/2):int(2*npts/3), :] = np.array([np.ones(int(npts/6))*-75 ,np.linspace(-250,-75,int(npts/6))]).T
	#right vertical
	pp1[int(2*npts/3):int(5*npts/6), :] = np.array([np.ones(int(npts/6))*75 ,np.linspace(-250,-75,int(npts/6))]).T
	#right horizontal
	pp1[int(5*npts/6):, :] = np.array([np.linspace(75,250,int(npts/6)),np.ones(int(npts/6))*-75]).T

	#rotate 90 deg
	# temp = np.array([pp1[:,1], pp1[:,0]]).T
	# pp1[:,:] = temp[:,:]

	pp2[:,:] = pp1[:,:]

	#transform scan2
	pp2 += t
	pp2 = rot.dot(pp2.T)
	pp2 = pp2.T

	#add noise to pp1 and pp2 SEPERATELY
	pp1 += np.random.randn(npts, 2)*noise_scale
	pp2 += np.random.randn(npts, 2)*noise_scale

	# ax.plot(pp1[:,0], pp1[:,1], color = (0.25,0.8,0.25,0.125), ls = '', marker = '.', markersize = 1)
	# ax.plot(pp2[:,0], pp2[:,1], color = (0.25,0.25,0.8,0.125), ls = '', marker = '.', markersize = 1)

	if output_actual == False:
		return pp1, pp2
	if output_actual == True:
		x_actual = np.array([t[0], t[1], theta])
		return pp1, pp2, x_actual

# def generate_along_track_data(fig,ax,draw = True, output_actual = False):

# 	""" Generates 10x10 unstructured scan for debugging """

# 	npts =  320 #16800
# 	tscale = 5
# 	noise_scale_x = 3 #debug notes: I am getting equal performance in x and y when I double x noise... (predicted error in x is still double predicted in y tho)
# 	noise_scale_y = 3

# 	pp1 = np.zeros([npts,2])
# 	pp2 = np.zeros([npts,2])


# 	#ADJUST NET ROTATION HERE
# 	theta = np.random.randn()*0.05 #0.1
# 	t = np.random.randn(2)*tscale
# 	# theta = 0.1
# 	# t = np.array([5,10])

# 	rot = R(theta)

# 	xctrs = [-175, -75, 75 , 175]
# 	yctrs = [-175, -75, 75, 175]

# 	count = 1
# 	for i in range(4):
# 		for j in range(4):

# 			pp1[(count-1)*(npts//16):(count)*(npts//16)] = np.array([xctrs[i],yctrs[j]])
# 			count += 1

# 	#rotate 90 deg
# 	# temp = np.array([pp1[:,1], pp1[:,0]]).T
# 	# pp1[:,:] = temp[:,:]

# 	# r2 = R(np.pi/3)
# 	# pp1 = r2.dot(pp1.T)
# 	# pp1 = pp1.T

# 	pp2[:,:] = pp1[:,:]

# 	#add noise to pp1 and pp2 SEPERATELY
# 	pp1[:,0] += np.random.randn(npts)*noise_scale_x
# 	pp1[:,1] += np.random.randn(npts)*noise_scale_y
# 	pp2[:,0] += np.random.randn(npts)*noise_scale_x
# 	pp2[:,1] += np.random.randn(npts)*noise_scale_y

# 	# pp2[:,:] = pp1[:,:]
# 	# pp2[:,0] += np.random.randn(npts)*0.01
# 	# pp2[:,1] += np.random.randn(npts)*0.01


# 	#transform scan2
# 	pp2 += t
# 	pp2 = rot.dot(pp2.T)
# 	pp2 = pp2.T

# 	# ax.plot(pp1[:,0], pp1[:,1], color = (0.25,0.8,0.25,0.125), ls = '', marker = '.', markersize = 1)
# 	# ax.plot(pp2[:,0], pp2[:,1], color = (0.25,0.25,0.8,0.125), ls = '', marker = '.', markersize = 1)

# 	if output_actual == False:
# 		return pp1, pp2
# 	if output_actual == True:
# 		x_actual = np.array([t[0], t[1], theta])
# 		return pp1, pp2, x_actual

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


		#was this
		# # standard deviations
		# std_x = np.sqrt(np.sum( (points[:,0] - mu_x)**2 ) / (np.shape(points)[0] - 1 ))
		# std_y = np.sqrt(np.sum( (points[:,1] - mu_y)**2 ) / (np.shape(points)[0] - 1 ))
		# #expected value
		# E = np.sum( (points[:,0] - mu_x) * (points[:,1] - mu_y) )/ (np.shape(points)[0] - 1)
		# # 	rho: pearson correlation coefficient
		# rho = E / (std_x*std_y)
		# sigma = np.array([[std_x**2,  rho*std_x*std_y],
		# 				  [-1*rho*std_x*std_y, std_y**2 ]])
		# print("sigma:",sigma)

		#test
		std_x = np.sum( (points[:,0] - mu_x)*(points[:,0] - mu_x) ) / (np.shape(points)[0] - 1 )
		std_y = np.sum( (points[:,1] - mu_y)*(points[:,1] - mu_y) ) / (np.shape(points)[0] - 1 )
		std_xy= np.sum((points[:,0] - mu_x)*(points[:,1] - mu_y) ) / (np.shape(points)[0] - 1)
		# std_yx= np.sum((points[:,1] - mu_y)*(points[:,0] - mu_x) ) / (np.shape(points)[0] - 1)
		sigma = np.array([[std_x,  std_xy],
						  [std_xy, std_y]])


		output = (mu, sigma)	
	return output

