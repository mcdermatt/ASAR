import numpy as np


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