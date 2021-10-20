import numpy as np
from vedo import *
import vtk
import os
# from numpy import sin, cos, tan
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
import time

def R(angs):
	"""generates rotation matrix using euler angles
	angs = np.array(phi, theta, psi) (aka rot about (x,y,z))"""

	phi = angs[0]
	theta = angs[1]
	psi = angs[2]

	mat = np.array([[cos(theta)*cos(psi), sin(psi)*cos(phi) + sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi)],
					[-sin(psi)*cos(theta), cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(psi) + sin(theta)*sin(psi)*cos(phi)],
					[sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta)]
						])
	return mat


def R_tf(angs):
	"""generates rotation matrix using euler angles
	angs = tf.constant(phi, theta, psi) (aka rot about (x,y,z))"""

	phi = angs[0]
	theta = angs[1]
	psi = angs[2]

	mat = tf.Variable([[cos(theta)*cos(psi), sin(psi)*cos(phi) + sin(phi)*sin(theta)*cos(psi), sin(phi)*sin(psi) - sin(theta)*cos(phi)*cos(psi)],
					[-sin(psi)*cos(theta), cos(phi)*cos(psi) - sin(phi)*sin(theta)*sin(psi), sin(phi)*cos(psi) + sin(theta)*sin(psi)*cos(phi)],
					[sin(theta), -sin(phi)*cos(theta), cos(phi)*cos(theta)]
						])
	return mat

def jacobian(angs, p_point):
	"""calculates jacobian for point using numpy
		angs = np.array(phi, theta, psi) aka (x,y,z)"""
	phi = angs[0]
	theta = angs[1]
	psi = angs[2]
	J = np.zeros([3,6])
	J[:3,:3] = np.identity(3)
	# (deriv of R() wrt phi).dot(p_point)
	J[:3,3] = np.array([[0., -sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi), cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi)],
						[0., -sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi), cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi)], 
						[0., -cos(phi)*cos(theta), -sin(phi)*cos(theta) ] ]).dot(p_point)
	# (deriv of R() wrt theta).dot(p_point)
	J[:3,4] = np.array([[-sin(theta)*cos(psi), cos(theta)*sin(phi)*cos(psi), -cos(theta)*cos(phi)*cos(psi)],
						[sin(psi)*sin(theta), -cos(theta)*sin(phi)*sin(psi), cos(theta)*sin(psi)*cos(phi)],
						[cos(theta), sin(phi)*sin(theta), -sin(theta)*cos(phi)]
						]).dot(p_point)
	J[:3,5] = np.array([[-cos(theta)*sin(psi), cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi), cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi) ],
						[-cos(psi)*cos(theta), -sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi), -sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi)],
						[0.,0.,0.]
						]).dot(p_point)

	return J

def jacobian_tf(angs, p_point):
	"""calculates jacobian for point using TensorFlow
		angs = tf.constant(phi, theta, psi) aka (x,y,z)"""

	phi = angs[0]
	theta = angs[1]
	psi = angs[2]

	#correct method using tf.tile
	eyes = tf.tile(tf.eye(3), [tf.shape(phi)[0] , 1])

	# slow method with for loop
	# eyes = tf.eye(3)
	# for c in range(tf.shape(angs)[1] - 1):
	# 	eyes = tf.concat([eyes, tf.eye(3)], axis = 0)

	# alt method using tf.while_loop()
	# i0 = tf.constant(1)
	# m0 = tf.constant([[1., 0., 0.], [0., 1., 0.] , [0., 0., 1.] ])
	# c = lambda i, m: i < (tf.shape(phi)[0]) #condition
	# b = lambda i, m: [i+1, tf.concat([m, m0], axis=0)] #while loop body
	# m, eyes = tf.while_loop(
	# 	c, b, loop_vars=[i0, m0],
	# 	shape_invariants=None)

	# (deriv of R() wrt phi).dot(p_point)
	#	NOTE: any time sin/cos operator is used, output will be 1x1 instead of constant (not good)
	Jx = tf.tensordot(tf.Variable([[tf.constant(0.), (-sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi))[0], (cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi))[0]],
								   [tf.constant(0.), (-sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi))[0], (cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi))[0]], 
								   [tf.constant(0.), (-cos(phi)*cos(theta))[0], (-sin(phi)*cos(theta))[0]] ]), p_point, axes = 1)

	# (deriv of R() wrt theta).dot(p_point)
	Jy = tf.tensordot(tf.Variable([[(-sin(theta)*cos(psi))[0], (cos(theta)*sin(phi)*cos(psi))[0], (-cos(theta)*cos(phi)*cos(psi))[0]],
								   [(sin(psi)*sin(theta))[0], (-cos(theta)*sin(phi)*sin(psi))[0], (cos(theta)*sin(psi)*cos(phi))[0]],
								   [(cos(theta))[0], (sin(phi)*sin(theta))[0], (-sin(theta)*cos(phi))[0]] ]), p_point, axes = 1)

	Jtheta = tf.tensordot(tf.Variable([[(-cos(theta)*sin(psi))[0], (cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi))[0], (cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi))[0] ],
									   [(-cos(psi)*cos(theta))[0], (-sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi))[0], (-sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi))[0]],
									   [tf.constant(0.),tf.constant(0.),tf.constant(0.)]]), p_point, axes = 1)


	Jx_reshape = tf.reshape(tf.transpose(Jx), shape = (tf.shape(Jx)[0]*tf.shape(Jx)[1],1))
	Jy_reshape = tf.reshape(tf.transpose(Jy), shape = (tf.shape(Jy)[0]*tf.shape(Jy)[1],1))
	Jtheta_reshape = tf.reshape(tf.transpose(Jtheta), shape = (tf.shape(Jtheta)[0]*tf.shape(Jtheta)[1],1))

	J = tf.concat([eyes, Jx_reshape, Jy_reshape, Jtheta_reshape], axis = 1)
	
	return J

def R2Euler(mat):
	"""determines euler angles from euler rotation matrix"""

	R_sum = np.sqrt(( mat[0,0]**2 + mat[0,1]**2 + mat[1,2]**2 + mat[2,2]**2 ) / 2)

	phi = np.arctan2(-mat[1,2],mat[2,2])
	theta = np.arctan2(mat[0,2], R_sum)
	psi = np.arctan2(-mat[0,1], mat[0,0])

	angs = np.array([phi, theta, psi])
	return angs


def R2Euler_tf(mat):
	"""determines euler angles from euler rotation matrix using tensorflow framework"""
	#TODO: get this working with vectorized operations

	# print("mat \n", mat)
	# mat = tf.cast(mat, dtype= tf.float32) #debug
	# print("mat \n", mat)


	R_sum = tf.math.sqrt(( mat[0,0]**2 + mat[0,1]**2 + mat[1,2]**2 + mat[2,2]**2 ) / 2)

	phi = tf.math.atan2(-mat[1,2],mat[2,2])
	theta = tf.math.atan2(mat[0,2], R_sum)
	psi = tf.math.atan2(-mat[0,1], mat[0,0])

	# angs = [phi, theta, psi]
	# print(phi, theta, psi)
	# angs = tf.concat([phi,psi], axis = 0)
	angs = tf.concat([tf.cast(phi, tf.float32), tf.cast(theta, tf.float32), tf.cast(psi, tf.float32)], axis = 0)

	#TODO: fix this
	# angs = tf.reshape(tf.transpose(angs), [3, tf.shape(mat)[2]]) #not reordering correctly

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
					min_num_pts = 20, nstd = 2, draw_grid = True, show_pc = True):

	""" Subdivide point cloud into consistantly sized rectangular voxles. Outputs mean center and
		covariance matrix for each voxel

	pc = input point cloud
	plt = plotter object (from Vedo)
	disp = display structure containing everything else we want to display
	bounds = np.array([minx, maxx, miny, maxy, minz, maxz])
	fid = np.array([ncellsx, ncellsy, ncellsz])

	"""
	cloud = Points(pc, c = (1,1,1), alpha = 0.5)
	if show_pc:
		disp.append(cloud) #add point cloud object to viz
	E = [] #strucutre to hold mus, sigmas and npts per voxel

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

	total_time = 0

	#loop through each voxel
	for x in range(fid[0]):
		for y in range(fid[1]):
			for z in range(fid[2]):
				loop_start = time.time()
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
					# print(eigenval,"\n", eigenvec)

					#was this
					# a1 = eigenval[0]
					# a2 = eigenval[1]
					# a3 = eigenval[2]
					# ell = Ell(pos=(mu[0], mu[1], mu[2]), axis1 = 4*np.sqrt(a1), 
					# 	axis2 = 4*np.sqrt(a2), axis3 = 4*np.sqrt(a3), 
					# 	angs = (np.array([-R2Euler(eigenvec)[0], -R2Euler(eigenvec)[1], -R2Euler(eigenvec)[2] ])), c=(1,0.5,0.5), alpha=1, res=12)

					big = np.argwhere(eigenval == np.max(eigenval))
					middle = np.argwhere(eigenval == np.median(eigenval))
					small = np.argwhere(eigenval == np.min(eigenval))
					# print(eigenval[big], eigenval[middle], eigenval[small])
					a1 = eigenval[big]
					a2 = eigenval[middle]
					a3 = eigenval[small]
					ell = Ell(pos=(mu[0], mu[1], mu[2]), axis1 = 4*np.sqrt(a1), 
						axis2 = 4*np.sqrt(a2), axis3 = 4*np.sqrt(a3), 
						angs = (np.array([-R2Euler(eigenvec)[big], -R2Euler(eigenvec)[middle], -R2Euler(eigenvec)[small] ])), c=(1,0.5,0.5), alpha=1, res=12)


					disp.append(ell)

					E.append((mu, sigma, np.shape(within_box)[0]))
					loop_time = time.time() - loop_start
					total_time += loop_time

	#test- add random ellipsoids and add them to disp
	# for i in range(10):
	# 	ell = Ellipsoid(pos=(np.random.randn()*10, np.random.randn()*10, 
	# 	np.random.rand()*5), axis1=(1, 0, 0), axis2=(0, 2, 0), axis3=(np.random.rand(), np.random.rand(), np.random.rand()), 
	# 	c=(np.random.rand(), np.random.rand(), np.random.rand()), alpha=1, res=12)
	# 	disp.append(ell)

	plt.show(disp, "subdivide_scan", at=0) 
	print("took", total_time/ (fid[0]*fid[1]*fid[2]), "seconds per loop")


	return E


def subdivide_scan_tf(pc, plt, bounds = tf.constant([-50.,50.,-50.,50.,-10.,10.]), fid = tf.constant([10,10,3]), disp = [],
					min_num_pts = 20, nstd = 2, draw_grid = True, show_pc = True):

	cloud = Points(pc, c = (1,1,1), alpha = 0.5)
	if show_pc:
		disp.append(cloud) #add point cloud object to viz

	c = tf.convert_to_tensor(pc, np.float32)
	#establish grid cells
	startx = bounds[0].numpy()
	stopx = bounds[1].numpy()
	numx = fid[0].numpy() + 1
	edgesx = tf.linspace(startx, stopx, numx)
	xbins = tfp.stats.find_bins(c[:,0], edgesx)
	# print(xbins)
	starty = bounds[2].numpy()
	stopy = bounds[3].numpy()
	numy = fid[1].numpy() + 1
	edgesy = tf.linspace(starty, stopy, numy)
	ybins = tfp.stats.find_bins(c[:,1], edgesy)
	# print(ybins)
	startz = bounds[4].numpy()
	stopz = bounds[5].numpy()
	numz = fid[2].numpy() +1
	edgesz = tf.linspace(startz, stopz, numz)
	zbins = tfp.stats.find_bins(c[:,2], edgesz)
	#print(zbins)

	E = []

	# print("shape of cloud: \n", tf.shape(c))

	total_time = 0
	#subdivide points into grid
	for x in range(fid[0]):
		for y in range(fid[1]):
			for z in range(fid[2]):
				loop_start = time.time()
				#only do calculations if there are a sufficicently high number of points in the bin
				xin = tf.transpose(tf.where(xbins == x))#[:,0]
				# print("shape of xin \n", tf.shape(xin))
				#get index for the subset of points in y that are also within a the correct x direction
				# yin = tf.where(tf.gather(ybins, xin) == y)[:,0] #only want the first column of gather
				yin = tf.transpose(tf.where(ybins == y))#[:,0]
				# print("shape of yin \n", tf.shape(yin))
				# idx = tf.where(tf.gather(zbins, yin) == z)
				zin = tf.transpose(tf.where(zbins == z))#[:,0]
				# print(xin[:10])
				# print(yin[:10])
				# print(zin[:10])
				idx = tf.sets.intersection(xin, yin)
				idx = tf.sparse.to_dense(idx)

				idx = tf.sets.intersection(idx, zin)
				idx = tf.sparse.to_dense(idx)

				# print("shape of idx \n", tf.shape(idx))
				num_in_cell = tf.shape(idx)[1]

				# print(num_in_cell)
				if num_in_cell > min_num_pts:

					# print(tf.squeeze(tf.gather(c,idx))) #bug here?
					# print("shape of input for fit gaussian \n",tf.squeeze(tf.gather(c,idx)))
					mu, sigma = fit_gaussian_tf(tf.squeeze(tf.gather(c,idx)))

					eig = tf.linalg.eig(sigma)
					eigenval = tf.cast(eig[0], tf.float32)
					eigenvec = tf.cast(eig[1], tf.float32)
					#for some reason TF likes to output eigs the opposite way as np...
					eigenval = tf.reverse(eigenval, axis = [0])
					eigenvec = tf.reverse(eigenvec, axis = [1])
					# print(eigenval, eigenvec)


					a1 = 4*tf.math.sqrt(eigenval[0])
					a2 = 4*tf.math.sqrt(eigenval[1])
					a3 = 4*tf.math.sqrt(eigenval[2])

					# print(a1)
					# print(a2)
					# print(a3)
					# print(mu)

					ell = Ell(pos=(mu[0], mu[1], mu[2]), axis1 = a1, 
						axis2 = a2, axis3 = a3, 
						angs = ([-R2Euler_tf(eigenvec)[0], -R2Euler_tf(eigenvec)[1], -R2Euler_tf(eigenvec)[2] ]), c=(1,0.5,0.5), alpha=1, res=12)

					disp.append(ell)

					E.append((mu, sigma, num_in_cell))

					loop_time = time.time() - loop_start
					total_time += loop_time
			
							# print(E)
	plt.show(disp, "subdivide_scan", at=0) 

	print("took", total_time/ (fid[0].numpy()*fid[1].numpy()*fid[2].numpy()), "seconds per loop")

	return E

class Ell(Mesh):
    """
    Build a 3D ellipsoid centered at position `pos`.

    |projectsphere|

    |pca| |pca.py|_
    """
    def __init__(self, pos=(0, 0, 0), axis1= 1, axis2 = 2, axis3 = 3, angs = np.array([0,0,0]),
                 c="cyan4", alpha=1, res=24):

        self.center = pos
        self.va_error = 0
        self.vb_error = 0
        self.vc_error = 0
        self.axis1 = axis1
        self.axis2 = axis2
        self.axis3 = axis3
        self.nr_of_points = 1 # used by pcaEllipsoid

        if utils.isSequence(res):
            res_t, res_phi = res
        else:
            res_t, res_phi = 2*res, res

        elliSource = vtk.vtkSphereSource()
        elliSource.SetThetaResolution(res_t)
        elliSource.SetPhiResolution(res_phi)
        elliSource.Update()
        l1 = axis1
        l2 = axis2
        l3 = axis3
        self.va = l1
        self.vb = l2
        self.vc = l3
        axis1 = 1
        axis2 = 1
        axis3 = 1
        angle = angs[0] #np.arcsin(np.dot(axis1, axis2))
        theta = angs[1] #np.arccos(axis3[2])
        phi =  angs[2] #np.arctan2(axis3[1], axis3[0])

        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Scale(l1, l2, l3)
        t.RotateX(np.rad2deg(angle))
        t.RotateY(np.rad2deg(theta))
        t.RotateZ(np.rad2deg(phi))
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputData(elliSource.GetOutput())
        tf.SetTransform(t)
        tf.Update()
        pd = tf.GetOutput()
        self.transformation = t

        Mesh.__init__(self, pd, c, alpha)
        self.phong()
        self.GetProperty().BackfaceCullingOn()
        self.SetPosition(pos)
        self.Length = -np.array(axis1) / 2 + pos
        self.top = np.array(axis1) / 2 + pos
        self.name = "Ell"


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

def fit_gaussian_tf(points):
	"""input: [N,3] tensor"""

	#TODO: remove elements from tensors that are all zeros -> need them to convert from raggedtensor type

	x = tf.math.reduce_mean(points[:,:,0], axis = 1)
	y = tf.math.reduce_mean(points[:,:,1], axis = 1)
	z = tf.math.reduce_mean(points[:,:,2], axis = 1)
	mu = tf.transpose(tf.Variable([x, y, z]))
	# print("mu: \n", mu)


	# print("pointsx: \n" , points[:,:,0])
	# print("mux: \n", mu[:,0])

	# print("pointsx - mux \n", tf.transpose(points[:, :, 0]) - mu[:,0])

	# print("\n tf.math.reduce_sum(points[:,0] - mu[0]) \n", tf.math.reduce_sum((points[:,0] - mu[0])))

	#standard deviations
	std_x = tf.math.reduce_sum( (tf.transpose(points[:,:,0]) - mu[:,0])**2, axis = 0) / tf.shape(points)[0].numpy()
	std_y = tf.math.reduce_sum( (tf.transpose(points[:,:,1]) - mu[:,1])**2, axis = 0) / tf.shape(points)[0].numpy()
	std_z = tf.math.reduce_sum( (tf.transpose(points[:,:,2]) - mu[:,2])**2, axis = 0) / tf.shape(points)[0].numpy()

	print("std_x \n",std_x)
	print("std_y \n",std_y)
	print("std_z \n",std_z)

	E_xy = tf.math.reduce_mean( (tf.transpose(points[:,:,0]) - mu[:,0]) * ( tf.transpose(points[:,:,1]) - mu[:,1]), axis =0 ) 	#expected value
	E_xz = tf.math.reduce_mean( (tf.transpose(points[:,:,0]) - mu[:,0]) * ( tf.transpose(points[:,:,2]) - mu[:,2]), axis =0 )
	E_yz = tf.math.reduce_mean( (tf.transpose(points[:,:,1]) - mu[:,1]) * ( tf.transpose(points[:,:,2]) - mu[:,2]), axis =0 )	

	# print("E_xy: \n", E_xy)

	sigma = tf.Variable([[std_x, E_xy, E_xz],
					     [E_xy, std_y, E_yz],
					     [E_xz, E_yz, std_z]])

	return mu, sigma