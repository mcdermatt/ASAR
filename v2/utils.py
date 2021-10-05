import numpy as np
from vedo import *
import os
from numpy import sin, cos, tan

def R(angs):
	"""generates rotation matrix using euler angles
	angs = np.array(phi, theta, psi) aka (x,y,z) """

	phi = angs[0]
	theta = angs[1]
	psi = angs[2]


	mat = np.array([[cos(theta)*cos(psi), sin(psi)*cos(phi) + sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi)],
					[-sin(psi)*cos(theta), cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(psi) + sin(theta)*sin(psi)*cos(phi)],
					[sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta)]
						])

	return mat

def jacobian(angs, p_point):
	"""calculates jacobian for point
		angs = np.array(phi, theta, psi) aka (x,y,z)"""

	phi = angs[0]
	theta = angs[1]
	psi = angs[2]

	J = np.zeros([3,6])
	J[:3,:3] = np.identity(3)

	# (deriv of R() wrt phi).dot(p_point)
	J[:3,3] = np.array([[0, -sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi), cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi)],
						[0, -sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi)], 
						[0, -cos(phi)*cos(theta), -sin(phi)*cos(theta) ] ]).dot(p_point)

	# (deriv of R() wrt theta).dot(p_point)
	J[:3,4] = np.array([[-sin(theta)*cos(psi), cos(theta)*sin(phi)*cos(psi), -cos(theta)*cos(phi)*cos(psi)],
						[sin(psi)*sin(theta), -cos(theta)*sin(phi)*sin(psi), cos(theta)*sin(psi)*cos(phi)],
						[cos(theta), sin(phi)*sin(theta), -sin(theta)*cos(phi)]
						]).dot(p_point)

	J[:3,5] = np.array([[-cos(theta)*sin(psi), cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi), cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi) ],
						[-cos(psi)*cos(theta), -sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi), -sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi)],
						[0,0,0]
						]).dot(p_point)

	print(J)

	return J

def R2Euler(mat):
	"""determines euler angles from euler rotation matrix"""

	R_sum = np.sqrt( mat[0,0]**2 + mat[0,1]**2 + mat[1,2]**2 + mat[2,2]**2 ) / 2

	phi = np.arctan2(-mat[1,2],mat[2,2])
	theta = np.arctan2(mat[0,2], R_sum)
	psi = np.arctan2(-mat[0,1], mat[0,0])

	angs = np.array([phi, theta, psi])
	return angs


def R_simp(n_hat, theta):
	"""Geneartes rotation matrix for rotation theta about axis n_hat using simple rotations
	n_hat = np.array([n1, n2, n3]), 
			n1^2 + n2^2 + n3^2 = 1
	theta = rotation (deg)	
	"""
	#test and make sure n_hat is a unit vector
	if np.sum((n_hat**2)) == 1:
		n1 = n_hat[0]
		n2 = n_hat[1]
		n3 = n_hat[2]
	#if not make it unit length
	else:
		print("not unit vec")
		n_hat_new = n_hat/np.linalg.norm(n_hat)
		n1 = n_hat_new[0]
		n2 = n_hat_new[0]
		n3 = n_hat_new[0]		
	mat = np.array([[cos(theta) + n1**2*(1-cos(theta)),  n1*n2*(1-cos(theta)) - n3*sin(theta), n1*n3*(1-cos(theta)) + n2*sin(theta)],
					[n1*n2*(1-cos(theta)) + n3*sin(theta), cos(theta) + n2**2*(1-cos(theta)), n2*n3*(1 - cos(theta)) - n1*sin(theta)], 
					[n1*n3*(1-cos(theta)) - n2*sin(theta), n2*n3*(1-cos(theta)) + n1*sin(theta), cos(theta) + n3**2*(1-cos(theta))]])
	return mat

def dR_simp(n_hat, theta):
	"""returns the derivative of the rotation matrix wrt theta, calculated from n_hat and theta using simple rotations
	https://arxiv.org/ftp/arxiv/papers/1311/1311.6010.pdf """
	#test and make sure n_hat is a unit vector
	if np.sum((n_hat**2)) == 1:
		n1 = n_hat[0]
		n2 = n_hat[1]
		n3 = n_hat[2]
	#if not make it unit length
	else:
		print("not unit vec")
		n_hat_new = n_hat/np.linalg.norm(n_hat)
		n1 = n_hat_new[0]
		n2 = n_hat_new[0]
		n3 = n_hat_new[0]		
	#get rotation matrix of the underlying rotation 
	R_mat = R_simp(n_hat, theta)
	#define skew-symmetric matrix S
	#	skew-symmetric defined as matrix where:
	#		A.T == -A
	# w == (dtheta/dt) = [[dtheta_x/dt],
	#					  [dtheta_y/dt],
	#					  [dtheta_z/dt]]
	# Sw = np.array([[0, -wz, wy],
	# 			   [wz, 0, -wx],
	# 			   [-wy, wx, 0]])
	#combine S and R_mat to get final derivative of rotation matrix
	# (dR/dt) = S(w)*R
	# mat = Sw.dot(R_mat) #don't need to use time derivative here
	# get deriviative of rotation matrix wrt theta
	S = np.array([[-sin(theta) + n1**2*sin(theta), n1*n2*sin(theta) - n3*cos(theta), n1*n3*sin(theta) - n2*cos(theta) ],
				  [n2*n2*sin(theta) + n3*cos(theta), -sin(theta) + n2**2*sin(theta), n2*n3*sin(theta) - n1*cos(theta) ],
				  [n1*n3*sin(theta) - n2*cos(theta), n2*n3*sin(theta) + n1*cos(theta), -sin(theta) + n3**2 * sin(theta)]]).dot(R_mat.T)
	mat = S.dot(R_mat)
	return mat

def subdivide_scan(pc, plt, bounds = np.array([-50,50,-50,50,-10,10]), fid = np.array([10,10,3]), disp = [],
					min_num_pts = 20, nstd = 2, draw_grid = True):

	""" Subdivide point cloud into consistantly sized rectangular voxles. Outputs mean center and
		covariance matrix for each voxel

	pc = input point cloud
	plt = plotter object (from Vedo)
	disp = display structure containing everything else we want to display
	bounds = np.array([minx, maxx, miny, maxy, minz, maxz])
	fid = np.array([ncellsx, ncellsy, ncellsz])

	"""
	cloud = Points(pc, c = (1,1,1), alpha = 0.5)
	disp.append(cloud) #add point cloud object to viz

	# draw divisions between voxel cells
	# b = shapes.Box(size=(bounds), c='g4', alpha=1) #meh

	xbound = np.linspace(bounds[0], bounds[1], fid[0] + 1)
	ybound = np.linspace(bounds[2], bounds[3], fid[1] + 1)
	zbound = np.linspace(bounds[4], bounds[5], fid[2] + 1)

	if draw_grid == True:
		for y in range(fid[1]+1):
			for z in range(fid[2]+1):
				p0 = np.array([xbound[-1], ybound[y], zbound[z]])
				p1 = np.array([xbound[0], ybound[y], zbound[z]])
				x_lines = shapes.Line(p0, p1, closed=False, c='white', alpha=1, lw=0.25, res=0)
				disp.append(x_lines)
		for x in range(fid[0]+1):
			for z in range(fid[2]+1):
				p0 = np.array([xbound[x], ybound[-1], zbound[z]])
				p1 = np.array([xbound[x], ybound[0], zbound[z]])
				y_lines = shapes.Line(p0, p1, closed=False, c='white', alpha=1, lw=0.25, res=0)
				disp.append(y_lines)
		for x in range(fid[0]+1):
			for y in range(fid[1]+1):
				p0 = np.array([xbound[x], ybound[y], zbound[-1]])
				p1 = np.array([xbound[x], ybound[y], zbound[0]])
				z_lines = shapes.Line(p0, p1, closed=False, c='white', alpha=1, lw=0.25, res=0)
				disp.append(z_lines)


	#loop through each voxel
	for x in range(fid[0]):
		for y in range(fid[1]):
			for z in range(fid[2]):
				within_x = pc[pc[:,0] > xbound[x]]
				within_x = within_x[within_x[:,0] < xbound[x+1] ]

				within_y = within_x[within_x[:,1] > ybound[y]]
				within_y = within_y[within_y[:,1] < ybound[y+1]]

				within_z = within_y[within_y[:,2] > zbound[z]]
				within_box = within_z[within_z[:,2] < zbound[z+1]]

				if np.shape(within_box)[0] > min_num_pts-1:
					mu, sigma = fit_gaussian(within_box)
					# print(sigma)
					eig = np.linalg.eig(sigma)
					eigenval = eig[0] #correspond to lengths of axis
					eigenvec = eig[1]

					# print(eigenvec)
					# print(eigenval)
					ans = nstd*np.sqrt(abs(eigenval.dot(eigenvec)))
					# if (ans < 8).all():
					# thresh = 4
					# temp = ans[ans < thresh] 
					# if np.shape(temp)[0] > 1: #dont want to draw stuff elongated in >1 dir
					ell = Ellipsoid(pos=(mu[0], mu[1], mu[2]), axis1=(ans[0], 0, 0), axis2=(0, ans[1], 0), axis3=(0,0,ans[2]), 
						c=(1,0.5,0.5), alpha=1, res=12)
					disp.append(ell)


	#test- add random ellipsoids and add them to disp
	# for i in range(10):
	# 	ell = Ellipsoid(pos=(np.random.randn()*10, np.random.randn()*10, 
	# 	np.random.rand()*5), axis1=(1, 0, 0), axis2=(0, 2, 0), axis3=(np.random.rand(), np.random.rand(), np.random.rand()), 
	# 	c=(np.random.rand(), np.random.rand(), np.random.rand()), alpha=1, res=12)
	# 	disp.append(ell)

	plt.show(disp, "subdivide_scan", at=0) 



def fit_gaussian(points):

	x = np.mean(points[:,0])
	y = np.mean(points[:,1])
	z = np.mean(points[:,2])
	mu = np.array([x, y, z])


	#standard deviations
	std_x = np.sqrt(np.sum( (points[:,0] - mu[0])**2 ) / np.shape(points)[0] )
	std_y = np.sqrt(np.sum( (points[:,1] - mu[1])**2 ) / np.shape(points)[0] )
	std_z = np.sqrt(np.sum( (points[:,2] - mu[2])**2 ) / np.shape(points)[0] )

	E_xy = np.mean( (points[:,0] - mu[0]) * (points[:,1] - mu[1]) ) 	#expected value
	E_xz = np.mean( (points[:,0] - mu[0]) * (points[:,2] - mu[2]) )
	E_yz = np.mean( (points[:,1] - mu[1]) * (points[:,2] - mu[2]) )	


	sigma = np.array([[std_x**2, E_xy, E_xz],
					  [E_xy, std_y**2, E_yz],
					  [E_xz, E_yz, std_z**2]])

	return mu, sigma