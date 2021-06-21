import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

#TODO
#	create list of ellipses 
#		use this to compare sequential scans

def ICP(E1,E2, num_cycles = 100):

	'''Iterative Closest Point algorithm:

	E1 = standard deviation ellipses from 1st scan
	E2 = standard deviation ellipses from 2nd scan'''

	#TODO- 
	#	check out point to plane metric- usually gives better results with imperfect points
	#		assumes poitns are sampled from real world surface
	#		uses least squares approach -> gaussian error metric?
	#	outliar rejection techniques	
	#	https://nbviewer.jupyter.org/github/niosus/notebooks/blob/master/icp.ipynb 

	for _ in num_cycles:

		#data association step

		#data alignment step
		pass

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

	#draw base point
	ax.plot(0,0,'ro', markersize = 10)

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