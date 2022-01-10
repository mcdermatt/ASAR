import numpy as np
from vedo import * #comment out on laptop
import vtk #comment out on laptop
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

def jacobian_tf(p_point, angs):
	"""calculates jacobian for point using TensorFlow
		angs = tf.constant[phi, theta, psi] aka (x,y,z)"""

	phi = angs[0]
	theta = angs[1]
	psi = angs[2]

	#correct method using tf.tile
	eyes = tf.tile(-tf.eye(3), [tf.shape(p_point)[1] , 1])

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
	#TODO: fix naming convention on these three vars - x, y, theta is inconsistant
	Jx = tf.tensordot(tf.Variable([[tf.constant(0.), (-sin(psi)*sin(phi) + cos(phi)*sin(theta)*cos(psi)), (cos(phi)*sin(psi) + sin(theta)*sin(phi)*cos(psi))],
								   [tf.constant(0.), (-sin(phi)*cos(psi) - cos(phi)*sin(theta)*sin(psi)), (cos(phi)*cos(psi) - sin(theta)*sin(psi)*sin(phi))], 
								   [tf.constant(0.), (-cos(phi)*cos(theta)), (-sin(phi)*cos(theta))] ]), p_point, axes = 1)

	# (deriv of R() wrt theta).dot(p_point)
	Jy = tf.tensordot(tf.Variable([[(-sin(theta)*cos(psi)), (cos(theta)*sin(phi)*cos(psi)), (-cos(theta)*cos(phi)*cos(psi))],
								   [(sin(psi)*sin(theta)), (-cos(theta)*sin(phi)*sin(psi)), (cos(theta)*sin(psi)*cos(phi))],
								   [(cos(theta)), (sin(phi)*sin(theta)), (-sin(theta)*cos(phi))] ]), p_point, axes = 1)

	Jz = tf.tensordot(tf.Variable([[(-cos(theta)*sin(psi)), (cos(psi)*cos(phi) - sin(phi)*sin(theta)*sin(psi)), (cos(psi)*sin(phi) + sin(theta)*cos(phi)*sin(psi)) ],
									   [(-cos(psi)*cos(theta)), (-sin(psi)*cos(phi) - sin(phi)*sin(theta)*cos(psi)), (-sin(phi)*sin(psi) + sin(theta)*cos(psi)*cos(phi))],
									   [tf.constant(0.),tf.constant(0.),tf.constant(0.)]]), p_point, axes = 1)

	# print(tf.shape(Jx))

	Jx_reshape = tf.reshape(tf.transpose(Jx), shape = (tf.shape(Jx)[0]*tf.shape(Jx)[1],1))
	Jy_reshape = tf.reshape(tf.transpose(Jy), shape = (tf.shape(Jy)[0]*tf.shape(Jy)[1],1))
	Jz_reshape = tf.reshape(tf.transpose(Jz), shape = (tf.shape(Jz)[0]*tf.shape(Jz)[1],1))

	#test 11/10
	# partials = tf.concat([Jx_reshape, Jy_reshape, Jz_reshape], axis = 1)
	# J = tf.concat([eyes,partials], axis = 1)

	J = tf.concat([eyes, Jx_reshape, Jy_reshape, Jz_reshape], axis = 1) #was this
	
	# print("\n J \n ", tf.shape(J))

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

# def subdivide_scan(pc, plt, bounds = np.array([-50,50,-50,50,-10,10]), fid = np.array([10,10,3]), disp = [],
# 					min_num_pts = 20, nstd = 2, draw_grid = True, show_pc = True):

# 	""" Subdivide point cloud into consistantly sized rectangular voxles. Outputs mean center and
# 		covariance matrix for each voxel

# 	pc = input point cloud
# 	plt = plotter object (from Vedo)
# 	disp = display structure containing everything else we want to display
# 	bounds = np.array([minx, maxx, miny, maxy, minz, maxz])
# 	fid = np.array([ncellsx, ncellsy, ncellsz])

# 	"""

# 	start = time.time()

# 	cloud = Points(pc, c = (1,1,1), alpha = 0.5)
# 	if show_pc:
# 		disp.append(cloud) #add point cloud object to viz
# 	E = [] #strucutre to hold mus, sigmas and npts per voxel

# 	# draw divisions between voxel cells
# 	# b = shapes.Box(size=(bounds), c='g4', alpha=1) #meh

# 	xbound = np.linspace(bounds[0], bounds[1], fid[0] + 1)
# 	ybound = np.linspace(bounds[2], bounds[3], fid[1] + 1)
# 	zbound = np.linspace(bounds[4], bounds[5], fid[2] + 1)

# 	if draw_grid == True:
# 		for y in range(fid[1]+1):
# 			for z in range(fid[2]+1):
# 				p0 = np.array([xbound[-1], ybound[y], zbound[z]])
# 				p1 = np.array([xbound[0], ybound[y], zbound[z]])
# 				x_lines = shapes.Line(p0, p1, closed=False, c='white', alpha=1, lw=0.25, res=0)
# 				disp.append(x_lines)
# 		for x in range(fid[0]+1):
# 			for z in range(fid[2]+1):
# 				p0 = np.array([xbound[x], ybound[-1], zbound[z]])
# 				p1 = np.array([xbound[x], ybound[0], zbound[z]])
# 				y_lines = shapes.Line(p0, p1, closed=False, c='white', alpha=1, lw=0.25, res=0)
# 				disp.append(y_lines)
# 		for x in range(fid[0]+1):
# 			for y in range(fid[1]+1):
# 				p0 = np.array([xbound[x], ybound[y], zbound[-1]])
# 				p1 = np.array([xbound[x], ybound[y], zbound[0]])
# 				z_lines = shapes.Line(p0, p1, closed=False, c='white', alpha=1, lw=0.25, res=0)
# 				disp.append(z_lines)

# 	mus = []
# 	sigmas = []
# 	sizes_list = []

# 	#loop through each voxel
# 	for x in range(fid[0]):
# 		for y in range(fid[1]):
# 			for z in range(fid[2]):
# 				within_x = pc[pc[:,0] > xbound[x]]
# 				within_x = within_x[within_x[:,0] < xbound[x+1] ]

# 				within_y = within_x[within_x[:,1] > ybound[y]]
# 				within_y = within_y[within_y[:,1] < ybound[y+1]]

# 				within_z = within_y[within_y[:,2] > zbound[z]]
# 				within_box = within_z[within_z[:,2] < zbound[z+1]]

# 				if np.shape(within_box)[0] > min_num_pts-1:
# 					mu, sigma = fit_gaussian(within_box)
# 					# print(mu)
# 					# print(sigma)
# 					eig = np.linalg.eig(sigma)
# 					eigenval = eig[0] #correspond to lengths of axis
# 					eigenvec = eig[1]
# 					# print(eigenval,"\n", eigenvec)

# 					##was this
# 					a1 = eigenval[0]
# 					a2 = eigenval[1]
# 					a3 = eigenval[2]
# 					ell = Ell(pos=(mu[0], mu[1], mu[2]), axis1 = 4*np.sqrt(a1), 
# 						axis2 = 4*np.sqrt(a2), axis3 = 4*np.sqrt(a3), 
# 						angs = (np.array([-R2Euler(eigenvec)[0], -R2Euler(eigenvec)[1], -R2Euler(eigenvec)[2] ])), c=(1,0.5,0.5), alpha=1, res=12)

# 					disp.append(ell)


# 					# # more consistant eigenvalue orders (similar to TF implementation) but worse(?) performance
# 					# big = np.argwhere(eigenval == np.max(eigenval))
# 					# middle = np.argwhere(eigenval == np.median(eigenval))
# 					# small = np.argwhere(eigenval == np.min(eigenval))
# 					# # print(eigenval[big], eigenval[middle], eigenval[small])
# 					# a1 = eigenval[big]
# 					# a2 = eigenval[middle]
# 					# a3 = eigenval[small]

# 					# ell2 = Ell(pos=(mu[0], mu[1], mu[2]), axis1 = 4*np.sqrt(a1), 
# 					# 	axis2 = 4*np.sqrt(a2), axis3 = 4*np.sqrt(a3), 
# 					# 	angs = (np.array([-R2Euler(eigenvec)[big], -R2Euler(eigenvec)[middle], -R2Euler(eigenvec)[small] ])), c=(0.5,0.5,1), alpha=1, res=12)

# 					# disp.append(ell2)

# 					# E.append([mu, sigma, np.shape(within_box)[0]])

# 					mus.append(mu)
# 					sigmas.append(sigma)
# 					sizes_list.append(np.shape(within_box)[0])

# 	plt.show(disp, "subdivide_scan", at=0) 
# 	print("took", time.time() - start, "seconds with numpy")


# 	# return E
# 	return(mus, sigmas, sizes_list)


def subdivide_scan_tf(cloud_tensor, plt, bounds = tf.constant([-50.,50.,-50.,50.,-10.,10.]), fid = tf.constant([10,10,3]), disp = [],
					min_num_pts = 20, draw = False, nstd = 2, draw_grid = True, draw_ell = True, show_pc = True):

	"""Subdivide point cloud into voxels and calculate means and covaraince matrices for each voxel
			cloud_tensor = point cloud input
			bounds = x, y, z lims 
			fid = number of voxel cells in x y and z
			disp = existing scene objects to draw on top of
			min_num_pts = only perform operations on voxels containing this many points or greater

	TODO: fix bug where bins get messed up if points exist outside bounds

			"""
	if show_pc == 1:
		color = (0.5,0.5,1)
	if show_pc == 2:
		color = (1,0.5,0.5)
	cloud = Points(cloud_tensor, c = color, alpha = 0.5)
	disp.append(cloud) #add point cloud object to viz

	# cloud_tensor = tf.convert_to_tensor(pc, np.float32)
	# print("cloud_tensor: \n", cloud_tensor)

	start = time.time()

	#establish grid cells
	startx = bounds[0].numpy()
	stopx = bounds[1].numpy()
	numx = fid[0].numpy() + 1
	edgesx = tf.linspace(startx, stopx, numx)
	xbins = tfp.stats.find_bins(cloud_tensor[:,0], edgesx)
	# print(xbins)
	starty = bounds[2].numpy()
	stopy = bounds[3].numpy()
	numy = fid[1].numpy() + 1
	edgesy = tf.linspace(starty, stopy, numy)
	ybins = tfp.stats.find_bins(cloud_tensor[:,1], edgesy)
	# print(ybins)
	startz = bounds[4].numpy()
	stopz = bounds[5].numpy()
	numz = fid[2].numpy() +1
	edgesz = tf.linspace(startz, stopz, numz)
	zbins = tfp.stats.find_bins(cloud_tensor[:,2], edgesz)
	#print(zbins)

	#create [N,3] query tensor containing the indices of all N bins
	fida = fid[0].numpy() 
	fidb = fid[1].numpy()
	fidc = fid[2].numpy()
	a = tf.linspace(0,fida-1,fida)[:,None]
	b = tf.linspace(0,fidb-1,fidb)[:,None]
	c = tf.linspace(0,fidc-1,fidc)[:,None]
	ansa = tf.tile(a, [fidb*fidc, 1])
	ansb = tf.tile(tf.reshape(tf.tile(b, [1,fida]), [-1,1] ), [(fidc), 1])
	ansc = tf.reshape(tf.tile(c, [1,fida*fidb]), [-1,1] )
	q = tf.cast(tf.squeeze(tf.transpose(tf.Variable([ansa,ansb,ansc]))), tf.float32)[:,None]
	# print("query: \n", q) #correct format

	#estabilsh tensor containing all points rounded to the nearest bin
	bins = tf.transpose(tf.Variable([xbins, ybins, zbins]))
	# print("\n bins \n", bins) #correct format	

	#takes ~0.65s with (25,25,4) grid-----------------------------------------------
	# This is where the largest biggest bottleneck is
	# https://stackoverflow.com/questions/46644796/which-one-is-more-efficient-tf-where-or-element-wise-multiplication

	#loc outputs tensor of shape [N,2], where:
	#  [[voxel number, index of [x,y,z] in cloud that corresponds to bin #],
	#   [voxel number,index of [x,y,z] in cloud that corresponds to bin #]...]  

	# idx = tf.equal(bins, q)
	# loc = tf.where(tf.math.equal(tf.math.reduce_all(idx, axis = 2), True))
	# # print("idx: \n", idx)
	#-------------------------------------------------------------------------------

	# Use binned point xyz location as hash for bin number -------------------------
	num = tf.cast( ( bins[:,0] + fida*bins[:,1] + (fida*fidb)*bins[:,2] ), tf.int32)
	loc = tf.concat((num[:,None], tf.cast(tf.linspace(0, tf.shape(bins)[0] - 1, tf.shape(bins)[0]  )[:,None],
                                      dtype = tf.int32) ), axis = 1 )
	# print("\n loc \n", loc[-10:])
	#VERY slow (takes minutes) ----------------------------------------------------
	# loc = None
	# for i in range(tf.shape(bins)[0]):
	# 	for j in range(tf.shape(q)[0]):
	# 		if tf.reduce_all(bins[i] == q[j]):
	# 			print(j,i)
	# 			try:
	# 				loc = tf.concat((loc2, tf.constant([[j,i]])), axis = 0)
	# 			except:
	# 				loc = tf.constant([[j,i]])
	#-------------------------------------------------------------------------------

	#Need to "ungroup" so that we can fit_gaussian_tf() to each individual voxel...
	s = tf.shape(loc)
	group_ids, group_idx = tf.unique(loc[:, 0], out_idx=s.dtype)
	# num_groups = tf.reduce_max(group_idx) + 1
	# print(group_ids, group_idx, num_groups)
	sizes = tf.math.bincount(group_idx)
	# print("\n sizes \n", sizes)

	# print("\n group_ids \n", group_ids)
	# print("\n group_idx \n", group_idx)
	# print("\n loc[:,0]: \n", loc)


	#BUG HERE? -> why am I getting idx 10 in tensor of length 10??? (bug was with linspace&cast2int)
	#sort loc by first element (group 0, group1, group2, etc )
	# print("\n loc: \n", loc)
	reordered = tf.argsort(loc, axis = 0, direction='ASCENDING')
	# print("\n reordered: \n", reordered)

	# print("\n applied to cloud ", tf.gather(cloud_tensor, reordered[:,0]))

	#updated sizes (order switches here)
	temp = tf.gather(loc, tf.argsort(loc[:,0], direction = "ASCENDING"))#[:,0]
	# print("\n temp \n", temp)
	sizes_updated = tf.math.bincount(temp[:,0])
	# print("sizes_updated \n", sizes_updated)

	# print("\n test of order \n", tf.gather(cloud_tensor, temp[:,1]))

	#replace <bins> here with <cloud_tensor> when done debugging
	# rag = tf.RaggedTensor.from_row_lengths(tf.gather(cloud_tensor, loc[:,0]), sizes) #was this

	#INDEXING ISSUE -> works on GPU, NOT CPU
	# test1 = tf.gather(cloud_tensor, temp[:,1])
	# print("\n test1 \n", test1)
	rag = tf.RaggedTensor.from_row_lengths(tf.gather(cloud_tensor, temp[:,1]), sizes_updated) 
	# print("ragged: \n", rag.bounding_shape())

	# #correct (ignores zeros) but way slower ---------------------------------------------
	# vox_with_zeros = rag.to_tensor() #[voxel #, point in voxel #, x y or z]
	# # print(vox_with_zeros)
	# sigma_slow = tfp.stats.covariance(vox_with_zeros[0,:sizes[0],:], sample_axis = 0, event_axis = 1)[:,None]
	# for i in range(tf.shape(vox_with_zeros)[0] - 1):
	# 	sigma_slow = tf.concat((sigma_slow, tfp.stats.covariance(vox_with_zeros[i,:sizes[i],:], sample_axis = 0, event_axis = 1)[:,None]), axis = 1 )
	# 	#TODO make sure this is concating in the correct order (1,2,3 vs 3,2,1...)
	# print(sigma_slow[:,1,:])
	# # # -----------------------------------------------------------------------------------

	mu = tf.math.reduce_mean(rag, axis=1) #works correctly
	#get rid of nan values in mu
	# print("\n mu before \n", mu)
	mu = tf.where(tf.math.is_nan(mu), tf.zeros_like(mu), mu) #was this
	# mu = tf.where(tf.math.is_nan(mu), -tf.ones_like(mu), mu) #test
	# print("\n mu after \n", mu)

	vox_with_zeros = rag.to_tensor() #[voxel #, point in voxel #, x y or z]
	# print(vox_with_zeros)

	#need to create mask vector to ignore all summation results from places where vox_with_zeros is 0,0,0
		#create a ragged tensor of ones using shape "sizes" and then convert to a standard tensor
	mask = tf.RaggedTensor.from_row_lengths(tf.ones(tf.math.reduce_sum(sizes_updated)), sizes_updated).to_tensor()
	# print("\n mask: \n", mask)

	#calculate mu manually (for debug)
	# mu_x = tf.math.reduce_sum(vox_with_zeros[:,:,0]*mask, axis = 1)/tf.cast(sizes_updated,tf.float32)
	# print(mu_x)

	# print("\n voxwzeros[:,:,0]: \n",vox_with_zeros[:,:,0])
	# print("\n mu[:,0] \n ", mu[:,0][:,None])
	# print("\n test2 \n", ( tf.reduce_sum( tf.math.square(vox_with_zeros[:,:,0]-mu[:,0][:,None])*mask,  axis = 1) )) #correct

	std_x = tf.reduce_sum( tf.math.square(vox_with_zeros[:,:,0]-mu[:,0][:,None])*mask , axis = 1)/ tf.cast(sizes_updated, tf.float32)
	std_y = tf.reduce_sum( tf.math.square(vox_with_zeros[:,:,1]-mu[:,1][:,None])*mask , axis = 1)/ tf.cast(sizes_updated, tf.float32)
	std_z = tf.reduce_sum( tf.math.square(vox_with_zeros[:,:,2]-mu[:,2][:,None])*mask , axis = 1)/ tf.cast(sizes_updated, tf.float32)

	# E_xy = mean((xpts-mux)(ypts-muy))
	E_xy = tf.reduce_sum( ((vox_with_zeros[:,:,0] - mu[:,0][:,None])*(vox_with_zeros[:,:,1] - mu[:,1][:,None]))*mask , axis = 1)/ tf.cast(sizes_updated, tf.float32)
	E_xz = -tf.reduce_sum( ((vox_with_zeros[:,:,0] - mu[:,0][:,None])*(vox_with_zeros[:,:,2] - mu[:,2][:,None]))*mask , axis = 1)/ tf.cast(sizes_updated, tf.float32)
	E_yz = -tf.reduce_sum( ((vox_with_zeros[:,:,1] - mu[:,1][:,None])*(vox_with_zeros[:,:,2] - mu[:,2][:,None]))*mask , axis = 1)/ tf.cast(sizes_updated, tf.float32)
	
	# [3,3,N]
	# sigma = tf.Variable([[std_x, E_xy, E_xz],
	# 					 [E_xy, std_y, E_yz],
	# 					 [E_xz, E_yz, std_z]]) 

	# [N, 3, 3]
	sigma = tf.Variable([std_x, E_xy, E_xz,
						 E_xy, std_y, E_yz,
						 E_xz, E_yz, std_z]) 
	sigma = tf.reshape(tf.transpose(sigma), (tf.shape(sigma)[1] ,3,3))
	# print("sigma \n", sigma)

	#get rid of any nan values in sigma

	sigma = tf.where(tf.math.is_nan(sigma), tf.zeros_like(sigma), sigma)
	# print(sigma)
	# sigma = tf.reshape(sigma, (tf.shape(sigma)[2] ,3,3))

	print("took", time.time() - start, "seconds with tensorflow")

	E = [mu, sigma, sizes, disp]
	if draw == True:

		disp = make_scene(plt, disp, E, color, bounds = bounds, draw_grid = draw_grid, draw_ell = draw_ell, fid = fid)


	# npts = sizes#was this
	npts = tf.gather(sizes_updated, tf.where(sizes_updated != 0))[:,0] #test

	E = [mu, sigma, npts, disp]

	return E

def make_scene(plt, disp, E, color, draw_grid = False, draw_ell = True, fid = None, bounds =None):

	"""draw distribution ellipses from E
	E = [mus, sigmas, sizes, disp]

	 called by subdivide_scan_tf() """

	mu = E[0]
	sigma = E[1]
	sizes = E[2] 

	# print(sigma)

	for i in range(tf.shape(sigma)[0]):

		eig = np.linalg.eig(sigma[i,:,:].numpy())
		eigenval = eig[0] #correspond to lengths of axis
		eigenvec = eig[1]
		# print("\n eigenvec \n" , eigenvec)
		# print("\n eivenval \n", eigenval)


		# eigenval, eigenvec = tf.linalg.eig(tf.transpose(sigma[:,:,i]))
		# eigenvec = tf.math.real(eigenvec)
		# eigenval = tf.math.real(eigenval)

		# a1 = eigenval[2]
		# a2 = eigenval[1]
		# a3 = eigenval[0]

		big = np.argwhere(eigenval == np.max(eigenval))[0,0]
		middle = np.argwhere(eigenval == np.median(eigenval))[0,0]
		smol = np.argwhere(eigenval == np.min(eigenval))[0,0]

		# a1 = eigenval[smol]
		# a2 = eigenval[middle] 
		# a3 = eigenval[big]

		# assmues decreasing size
		a1 = eigenval[0]
		a2 = eigenval[1]
		a3 = eigenval[2]

		# print(a1,a2,a3) #floats
		# print(mu)
		# print("\n eigenvec \n", eigenvec)
		# print("\n eigenval \n", eigenval)

		# print(R2Euler(eigenvec))

		if draw_ell == True:			
			if mu[i,0] != 0 and mu[i,1] != 0:
				ell = Ell(pos=(mu[i,0], mu[i,1], mu[i,2]), axis1 = 4*np.sqrt(abs(a1)), 
					axis2 = 4*np.sqrt(abs(a2)), axis3 = 4*np.sqrt(abs(a3)), 
					angs = (np.array([-R2Euler(eigenvec)[0], -R2Euler(eigenvec)[1], -R2Euler(eigenvec)[2] ])), c=color, alpha=1, res=12)
		#todo - fix rotation bug in angs[1]
				
				disp.append(ell)

	if draw_grid == True:

		xbound = np.linspace(bounds[0], bounds[1], fid[0] + 1)
		ybound = np.linspace(bounds[2], bounds[3], fid[1] + 1)
		zbound = np.linspace(bounds[4], bounds[5], fid[2] + 1)

		# #normal (for 3D)---------------------------------
		# for y in range(fid[1]+1):
		# 	for z in range(fid[2]+1):
		# 		p0 = np.array([xbound[-1], ybound[y], zbound[z]])
		# 		p1 = np.array([xbound[0], ybound[y], zbound[z]])
		# 		x_lines = shapes.Line(p0, p1, closed=False, c='black', alpha=1, lw=0.25, res=0)
		# 		disp.append(x_lines)
		# for x in range(fid[0]+1):
		# 	for z in range(fid[2]+1):
		# 		p0 = np.array([xbound[x], ybound[-1], zbound[z]])
		# 		p1 = np.array([xbound[x], ybound[0], zbound[z]])
		# 		y_lines = shapes.Line(p0, p1, closed=False, c='black', alpha=1, lw=0.25, res=0)
		# 		disp.append(y_lines)
		# for x in range(fid[0]+1):
		# 	for y in range(fid[1]+1):
		# 		p0 = np.array([xbound[x], ybound[y], zbound[-1]])
		# 		p1 = np.array([xbound[x], ybound[y], zbound[0]])
		# 		z_lines = shapes.Line(p0, p1, closed=False, c='black', alpha=1, lw=0.25, res=0)
		# 		disp.append(z_lines)
		# #-------------------------------------------------

		# for 2d viz -------------------------------------
		for y in range(fid[1]+1):
			for z in range(fid[2]):
				p0 = np.array([xbound[-1], ybound[y], 0])
				p1 = np.array([xbound[0], ybound[y], 0])
				x_lines = shapes.Line(p0, p1, closed=False, c='black', alpha=1, lw=0.25, res=0)
				disp.append(x_lines)
		for x in range(fid[0]+1):
			for z in range(fid[2]):
				p0 = np.array([xbound[x], ybound[-1], 0])
				p1 = np.array([xbound[x], ybound[0], 0])
				y_lines = shapes.Line(p0, p1, closed=False, c='black', alpha=1, lw=0.25, res=0)
				disp.append(y_lines)
		#-------------------------------------------------

	return(disp)
	# plt.show(disp, "subdivide_scan", at=0) #was here, moving to inside main loop


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
	"""DEPRECIATED
	input: [N,3] tensor"""

	# print("input to fit_gaussian_tf(): \n", points)

	x = tf.math.reduce_mean(points[:,:,0], axis = 1)
	y = tf.math.reduce_mean(points[:,:,1], axis = 1)
	z = tf.math.reduce_mean(points[:,:,2], axis = 1)
	mu = tf.transpose(tf.Variable([x, y, z]))


	# print("points: \n" , points)
	# print("mux: \n", mu[:,0])

	# print("pointsx - mux \n", tf.transpose(points[:, :, 0]) - mu[:,0])

	# print("\n tf.math.reduce_sum(points[:,0] - mu[0]) \n", tf.math.reduce_sum((points[:,0] - mu[0])))

	# nonzeroxidx = tf.math.not_equal(points, tf.constant([[0.,0.,0.]])) 
	# print(nonzeroxidx)
	# nonzerox = tf.squeeze(tf.gather(points,tf.where(tf.math.reduce_any(nonzeroxidx, axis = 2) == True)))
	# print(nonzerox)

	#standard deviations
	std_x = tf.math.reduce_sum( (tf.transpose(points[:,:,0]) - mu[:,0])**2, axis = 0) / tf.shape(points)[0].numpy()
	std_y = tf.math.reduce_sum( (tf.transpose(points[:,:,1]) - mu[:,1])**2, axis = 0) / tf.shape(points)[0].numpy()
	std_z = tf.math.reduce_sum( (tf.transpose(points[:,:,2]) - mu[:,2])**2, axis = 0) / tf.shape(points)[0].numpy()

	# print("std_x \n",std_x)
	# print("std_y \n",std_y)
	# print("std_z \n",std_z)

	E_xy = tf.math.reduce_mean( (tf.transpose(points[:,:,0]) - mu[:,0]) * ( tf.transpose(points[:,:,1]) - mu[:,1]), axis =0 ) 	#expected value
	E_xz = tf.math.reduce_mean( (tf.transpose(points[:,:,0]) - mu[:,0]) * ( tf.transpose(points[:,:,2]) - mu[:,2]), axis =0 )
	E_yz = tf.math.reduce_mean( (tf.transpose(points[:,:,1]) - mu[:,1]) * ( tf.transpose(points[:,:,2]) - mu[:,2]), axis =0 )	

	# print("E_xy: \n", E_xy)

	sigma = tf.Variable([[std_x, E_xy, E_xz],
					     [E_xy, std_y, E_yz],
					     [E_xz, E_yz, std_z]])

	return mu, sigma

def get_correspondences_tf(a, b, mu1, mu2, bounds, fid, method = "voxel", disp = None, draw_corr = False):
	"""finds closet point on b for each point in a
		
		pass in disp and set draw_corr to true to draw correspondences 
	
		includes m1 and m2 to for drawing correspondence arrows (keeps track of unused voxels)
	
	"""
	#TODO: fix bug that occurs when only one voxel is used in scan2
	#TODO: get rid of unnecessary operations in voxel method

	# print("\n corr shape a b \n", tf.shape(a), tf.shape(b))
	# print("a, b \n", a[:10], "\n", b[:10])

	if method == "NN":
		if tf.shape(tf.shape(a)) == 1:
			# print("\n before \n",a)
			a = a[None,:][None,:]
			# print(a)
		else:
			a = a[:,None]

		# using nearest neighbor (1-NN) -----------------------------------------
		dist = tf.math.reduce_sum( (tf.square( tf.math.subtract(a, b) ))  , axis = 2)
		# print("\n dist \n", dist)

		ans = tf.where( tf.transpose(dist) == tf.math.reduce_min(dist, axis = 1))
		# print("\n shortest dist \n", ans)

		reordered = tf.argsort(ans[:,1], axis = 0)
		corr = tf.gather(ans,reordered)
		# print("\n reordered \n", corr)
		#-----------------------------------------------------------------------
		# print("corr \n", corr)
		if draw_corr == True:
			for i in range(tf.shape(corr)[0]):
				pt1 = tf.squeeze(a[corr[i][1].numpy()])
				pt2 = tf.squeeze(b[corr[i][0].numpy()])
				# print("pt1", pt1)
				# arrow = shapes.Line(pt1.numpy(), pt2.numpy(), closed = False, c = 'white', lw = 4) #line
				arrow = shapes.Arrow(pt1.numpy(), pt2.numpy(), c = 'black')
				disp.append(arrow)
		#	[cell in b, cell in a]
			return(corr, disp)
		else:
			return corr

	if method == "voxel":

		#TODO: remove voxels in a with flagged indices from main loop (these have no corresponding
		#	    b distributions in the same voxel)

		#get voxel number of each distribution in a ----------------------------
		startx = bounds[0].numpy()
		stopx = bounds[1].numpy()
		numx = fid[0].numpy() + 1
		edgesx = tf.linspace(startx, stopx, numx)

		starty = bounds[2].numpy()
		stopy = bounds[3].numpy()
		numy = fid[1].numpy() + 1
		edgesy = tf.linspace(starty, stopy, numy)

		startz = bounds[4].numpy()
		stopz = bounds[5].numpy()
		numz = fid[2].numpy() + 1
		edgesz = tf.linspace(startz, stopz, numz)

		xbinsa = tfp.stats.find_bins(a[:,0], edgesx)
		ybinsa = tfp.stats.find_bins(a[:,1], edgesy)
		zbinsa = tfp.stats.find_bins(a[:,2], edgesz)

		binsa = tf.transpose(tf.Variable([xbinsa, ybinsa, zbinsa]))
		numa = tf.cast( ( binsa[:,0] + fid[0].numpy()*binsa[:,1] + (fid[0].numpy()*fid[1].numpy())*binsa[:,2] ), tf.int32)

		#get voxel number of each distribution in b --------------------------
		xbinsb = tfp.stats.find_bins(b[:,0], edgesx)
		ybinsb = tfp.stats.find_bins(b[:,1], edgesy)
		zbinsb = tfp.stats.find_bins(b[:,2], edgesz)

		binsb = tf.transpose(tf.Variable([xbinsb, ybinsb, zbinsb]))
		numb = tf.cast( ( binsb[:,0] + fid[0].numpy()*binsb[:,1] + (fid[0].numpy()*fid[1].numpy())*binsb[:,2] ), tf.int32)
		# print("\n numa \n", numa[:50])
		# print("\n numb \n", numb[:50])

		# find indices of voxels in b (if any) that match each element of a --
		numa = numa[:,None] #need to add axis to a so all this to be run in parallel

		# print("\n numa, numb \n ", numa, numb)

		eq = tf.cast(tf.where(numa == numb), tf.int32)
		#eq is correct BUT it much shorter than the origonal vec numa
		# print("\n voxels that have valid means from both scans \n", eq) #[idx_a, idx_b]

		# #create full length correspondence vec -----------------------------------
		# #set points of a with no point in b to -1 
		# # corr = tf.concat((tf.ones(tf.shape(numa)[0])))

		# #find which elements of a share a point on b
		# mask_match = tf.cast(tf.math.reduce_any(tf.math.equal(numa , numb), axis = 1), tf.int32)
		# # print(mask_match)
		# no_match = tf.squeeze(tf.where(mask_match == 0))
		# # print("\n no_match \n", no_match)

		# # print(tf.gather(numa, no_match))

		# #TODO -> get this to output index # not actual value		
		# #compile all unmatched distributions of a and -1
		# bads = tf.concat(((tf.cast(no_match, tf.int32))[:,None], 
		# 				   tf.cast(-1*tf.ones(tf.shape(no_match)[0]), tf.int32)[:,None] ), axis = 1)
		# # print(bads)

		# corr = tf.concat((eq, bads), axis = 0)
		# # print(corr)

		# order = tf.argsort(corr[:,0])
		# # print(order)
		# corr = tf.gather(corr, order)
		# #need to reverse axis
		# corr = tf.concat((corr[:,1][:,None],corr[:,0][:,None]), axis = 1)
		# #-----------------------------------------------------------------------------

		#switch order and only return elements that are useful
		corr = tf.concat((eq[:,1][:,None],eq[:,0][:,None]), axis = 1) #[b,a]
		# print("\n corr \n", corr[:10])
		# print("\n a \n",tf.shape(a))

		if draw_corr == True:
			for i in range(tf.shape(corr)[0]):
				#was this
				# pt1 = a[eq[i][0].numpy()]
				# pt2 = b[eq[i][1].numpy()]

				pt1 = a[corr[i,1].numpy()]
				pt2 = b[corr[i,0].numpy()]

				arrow = shapes.Line(pt1.numpy(), pt2.numpy(), closed = False, c = 'black', lw = 2) #line
				# arrow = shapes.Arrow(pt2.numpy(), pt1.numpy(), c = 'white')
				disp.append(arrow)

				#FOR DEBUG-> draw ell in center of each used voxel
				# ell_test = Ell(pos = pt1.numpy(), axis1 = 5, axis2 = 5, axis3 = 5, c = (i/(300), i/(300), i/(300) ))
				# disp.append(ell_test)
		#	[cell in b, cell in a]
			return(corr, disp)
		else:
			return corr


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

        #needed theta and angle to be negative before messing with E_xz, E_yz...
        t.RotateZ(np.rad2deg(phi))
        t.RotateY(np.rad2deg(theta))
        t.RotateX(np.rad2deg(angle))
        
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

#_______________________________________________________________________________________________

# def generate_test_dataset():
	
# 	""" Generate a simple 3D T shaped intersection """

# 	#TODO: take in transformation from pp1 to pp2

# 	bounds = tf.constant ([-150.,150.,-150.,150.,-150,150])
# 	# x = tf.constant([0.1, 3., 0.2, -0.02, -0.05, -0.12])
# 	# x = tf.constant([0., 0., 0., 0., 0., 0.])
# 	# x = tf.constant([0.1, 3., 0.2, -0.02, -0.05, -0.12])
# 	x = tf.constant([1., 3., 2., -0.12, -0.1, -0.2])
# 	# x = tf.constant([1., 3., 2., -0., 0.0, -0.2])


# 	height = 40
# 	hs = 1 #height spacing
# 	ns = 200 #number of points scale

# 	xpos = tf.linspace(-100., 100., ns)[:,None]
# 	ypos = -60*tf.ones(ns)[:,None]
# 	# ypos = (-60*tf.ones(ns) + 10*tf.sin(tf.linspace(-3., 3., ns)))[:,None]
# 	zpos = tf.ones(ns)[:,None]
# 	pp1 = tf.concat((xpos, ypos, zpos), axis = 1)

# 	for i in range(height*5):
# 		#back wall
# 		if i < height:
# 			xpos = tf.linspace(-100., 100., ns)[:,None]
# 			ypos = -60*tf.ones(ns)[:,None]
# 			# ypos = (-60*tf.ones(ns) + 10*tf.sin(tf.linspace(-3., 3., ns)))[:,None]
# 			zpos = hs*i*tf.ones(ns)[:,None]
# 			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

# 		if i > height and i < 2*height:
# 			ypos = tf.linspace(-20., 100., int(.6*ns))[:,None]
# 			xpos = -30*tf.ones(int(.6*ns))[:,None]
# 			zpos = hs*(i%height)*tf.ones(int(.6*ns))[:,None]
# 			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

# 		if i > 2*height and i < 3*height:
# 			ypos = tf.linspace(-20., 100., int(.6*ns))[:,None]
# 			xpos = 30*tf.ones(int(.6*ns))[:,None]
# 			zpos = hs*(i%height)*tf.ones(int(.6*ns))[:,None]
# 			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

# 		#left & right connector
# 		if i > 3*height and i < 4*height:
# 			xpos = tf.linspace(30., 100., int(.7*ns))[:,None]
# 			ypos = -20*tf.ones(int(.7*ns))[:,None]
# 			zpos = hs*(i%height)*tf.ones(int(.7*ns))[:,None]
# 			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)
# 		if i > 4*height and i < 5*height:
# 			xpos = tf.linspace(-30., -100., int(.7*ns))[:,None]
# 			ypos = -20*tf.ones(int(.7*ns))[:,None]
# 			zpos = hs*(i%height)*tf.ones(int(.7*ns))[:,None]
# 			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

# 		pp1 = tf.concat((pp1, pp1_i), axis = 0)

# 	#add floor -------------------------------------------
# 	for i in range(-100, 100, 1):
# 		ypos = tf.linspace(-60., -20., 100)[:,None]
# 		xpos = i*tf.ones(100)[:,None]
# 		zpos = tf.ones(100)[:,None]
# 		pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)
# 		pp1 = tf.concat((pp1, pp1_i), axis = 0)
# 	for i in range(-30, 30, 1):
# 		ypos = tf.linspace(-20., 100., 100)[:,None]
# 		xpos = i*tf.ones(100)[:,None]
# 		zpos = tf.ones(100)[:,None]
# 		pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)
# 		pp1 = tf.concat((pp1, pp1_i), axis = 0)
# 	#------------------------------------------------------
	
# 	print(tf.shape(pp1))
# 	# for debug - add particles in middle to prevent L1 rank deficiency bug
# 	# pp1 = tf.concat((pp1, tf.random.normal((100,3))), axis = 0)

# 	#rotate scan 1
# 	# pp1 = pp1 @ (R_tf(tf.constant([0.1,0.1,-0.1]))) + tf.constant([1.,2.,3.])

# 	# pp2 = tf.random.normal((100,3))
# 	rot = R_tf(x[3:])
# 	pp2 = pp1 @ rot + x[:3]

# 	#add a little bit of noise
# 	pp1 = pp1 + tf.random.normal(tf.shape(pp1))*0.2
# 	pp2 = pp2 + tf.random.normal(tf.shape(pp2))*0.2

# 	return(pp1, pp2, bounds, x)
#______________________________________________________________________________________________

# def generate_test_dataset():
	
# 	""" Generate a simple 2D tunnel scene """

# 	#TODO: take in transformation from pp1 to pp2

# 	bounds = tf.constant ([-150.,150.,-150.,150.,-150,150])
# 	x = tf.constant([1., 4., 2., -0., 0.0, -0.1])
# 	# x = tf.constant([tf.random.normal([1]).numpy(), tf.random.normal([1]).numpy(), 
#  #                  tf.random.normal([1]).numpy(), 0.03*tf.random.normal([1]).numpy(), 
#  #                  0.03*tf.random.normal([1]).numpy(), 0.1*tf.random.normal([1]).numpy()])
# 	# x = tf.squeeze(x)
# 	print("starting transformation \n", x)

# 	height = 40
# 	hs = 1 #height spacing
# 	ns = 500 #number of points scale
# 	sp = -100. #start point

# 	ypos = tf.linspace(sp, 100., int(.6*ns))[:,None]
# 	xpos = -30*tf.ones(int(.6*ns))[:,None]
# 	zpos = hs*(height)*tf.ones(int(.6*ns))[:,None]
# 	pp1 = tf.concat((xpos, ypos, zpos), axis = 1)

# 	for i in range(height*2):

# 		if i < height:
# 			ypos = tf.linspace(sp, 100., int(.6*ns))[:,None]
# 			xpos = -30*tf.ones(int(.6*ns))[:,None]
# 			zpos = hs*(i%height)*tf.ones(int(.6*ns))[:,None]
# 			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

# 		if i > height:
# 			ypos = tf.linspace(sp, 100., int(.6*ns))[:,None]
# 			xpos = 30*tf.ones(int(.6*ns))[:,None]
# 			zpos = hs*(i%height)*tf.ones(int(.6*ns))[:,None]
# 			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

# 		pp1 = tf.concat((pp1, pp1_i), axis = 0)

# 	#add floor -------------------------------------------
# 	for i in range(-30, 30, 1):
# 		ypos = tf.linspace(sp, 100., int(.6*ns))[:,None]
# 		xpos = i*tf.ones(int(.6*ns))[:,None]
# 		zpos = tf.ones(int(.6*ns))[:,None]
# 		pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)
# 		pp1 = tf.concat((pp1, pp1_i), axis = 0)
# 	#------------------------------------------------------
	
# 	print(tf.shape(pp1))

# 	#rotate scan 1
# 	# pp1 = pp1 @ (R_tf(tf.constant([0.1,0.1,-0.1]))) + tf.constant([1.,2.,3.])

# 	rot = R_tf(x[3:])
# 	# pp2 = pp1 @ rot + x[:3] #was this 11/30
# 	pp2 = (pp1 + x[:3]) @ rot #actually think this is better

# 	#add noise
# 	pp1 = pp1 + tf.random.normal(tf.shape(pp1))*0.2
# 	pp2 = pp2 + tf.random.normal(tf.shape(pp2))*0.2

# 	return(pp1, pp2, bounds, x)

#_____________________________________________________________________________________

# def generate_test_dataset():
	
# 	""" Generate a single wall"""

# 	#TODO: take in transformation from pp1 to pp2

# 	bounds = tf.constant ([-150.,150.,-150.,150.,-150,150])
# 	# x = tf.constant([0., 0., 0., 0., 0., 0.])
# 	# x = tf.constant([1., 3., 2., -0.12, -0.1, -0.2])
# 	x = tf.constant([0., 1., 0., 0., 0.0, -0.1])


# 	height = 120 #40
# 	hs = 1 #height spacing
# 	ns = 200 #number of points scale

# 	xpos = tf.linspace(-100., 100., ns)[:,None]
# 	ypos = -60*tf.ones(ns)[:,None]
# 	# ypos = (-60*tf.ones(ns) + 10*tf.sin(tf.linspace(-3., 3., ns)))[:,None]
# 	zpos = tf.ones(ns)[:,None]
# 	pp1 = tf.concat((xpos, ypos, zpos), axis = 1)

# 	for i in range(height):
# 		#back wall
# 		if i < height:
# 			xpos = tf.linspace(-100., 100., ns)[:,None]
# 			ypos = -60*tf.ones(ns)[:,None]
# 			# ypos = (-60*tf.ones(ns) + 10*tf.sin(tf.linspace(-3., 3., ns)))[:,None]
# 			zpos = hs*i*tf.ones(ns)[:,None]
# 			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

# 		pp1 = tf.concat((pp1, pp1_i), axis = 0)

	
# 	print(tf.shape(pp1))

# 	#rotate scan 1
# 	# pp1 = pp1 @ (R_tf(tf.constant([0.1,0.1,-0.1]))) + tf.constant([1.,2.,3.])
# 	# pp1 += tf.constant([1.,2.,3.])

# 	# pp2 = tf.random.normal((100,3))
# 	rot = R_tf(x[3:])
# 	pp2 = pp1 @ rot + x[:3]
# 	# pp2 = (pp1 + x[:3]) @ rot

# 	#add a little bit of noise
# 	pp1 = pp1 + tf.random.normal(tf.shape(pp1))*0.2
# 	pp2 = pp2 + tf.random.normal(tf.shape(pp2))*0.2


# 	return(pp1, pp2, bounds, x)

#____________________________________________________________________________________________

def generate_test_dataset():

	""" Generate FLAT tunnel scene for 2D-ICET paper"""
	tf.random.set_seed(1337)

	bounds = tf.constant([-15.,15.,-50.,50.,-1,1])
	x = tf.constant([1., 0., 0., -0., 0.0, 0.1]) #was this for demo
	# x = tf.constant([0.1, 0., 0., 0., 0., 0.01]) #used for last 

	print("starting transformation \n", x)

	height = 1
	hs = 1 #height spacing
	ns = 500 #number of points scale
	sp = -40. #start point


	for i in range(height*2):

		if i < height:
			ypos = tf.linspace(sp, 40., ns)[:,None]
			xpos = -5*tf.ones(ns)[:,None]
			zpos = hs*(i%height)*tf.ones(ns)[:,None]
			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

		if i >= height:
			ypos = tf.linspace(sp, 40., ns)[:,None]
			xpos = 5*tf.ones(ns)[:,None]
			zpos = hs*(i%height)*tf.ones(ns)[:,None]
			pp1_i = tf.concat((xpos, ypos, zpos), axis = 1)

		if i == 0:
			pp1 = pp1_i
		else:
			pp1 = tf.concat((pp1, pp1_i), axis = 0)

	
	print(tf.shape(pp1))

	#rotate scan 1
	# pp1 = pp1 @ (R_tf(tf.constant([0.1,0.1,-0.1]))) + tf.constant([1.,2.,3.])

	#adjust initial pc relative to car
	pp1 = pp1 + tf.constant([-3.,4.,0.0])

	rot = R_tf(x[3:])
	# pp2 = pp1 @ rot + x[:3] #was this 11/30
	pp2 = (pp1 + x[:3]) @ rot #actually think this is better

	#add noise
	scale = tf.constant([0.1,0.01,0.01])
	pp1 = pp1 + tf.random.normal(tf.shape(pp1))*scale
	pp2 = pp2 + tf.random.normal(tf.shape(pp2))*scale

	return(pp1, pp2, bounds, x)