import numpy as np
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget
import time
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from utils import R2Euler, Ell, jacobian_tf, R_tf

#TODO:
	# take stuff out of __init__, begin working on main loop
	# figure out why <get_occupied()> always includes cell 0 at the end
	# why are some arrows from visualize_L() not perfectly aligned with distribution axis??
	# for voxles that have an insufficient number of points, sweep through the corresponding spike until we find one with enough points
	# try stretching out voxels at steep angles
	# make easy way to turn visualization off to improve runtime

class ICET():

	def __init__(self, cloud1, cloud2, fid = 30, draw = True):

		self.min_cell_distance = 3 #begin closest spherical voxel here
		self.min_num_pts = 10 #ignore "occupied" cells with fewer than this number of pts
		self.fid = fid # dimension of 3D grid: [fid, fid, fid]
		self.draw = draw

		#convert cloud1 to tesnsor
		self.cloud1_tensor = tf.cast(tf.convert_to_tensor(cloud1), tf.float32)
		self.cloud2_tensor = tf.cast(tf.convert_to_tensor(cloud2), tf.float32)
		self.cloud2_tensor_OG = tf.cast(tf.convert_to_tensor(cloud2), tf.float32) #TODO- try and avoid repeating this...

		self.plt = Plotter(N = 1, axes = 4, bg = (1, 1, 1), interactive = True) #axis = 4
		self.disp = []

		#convert cloud to spherical coordinates
		self.cloud1_tensor_spherical = tf.cast(self.c2s(self.cloud1_tensor), tf.float32)
		self.cloud2_tensor_spherical = tf.cast(self.c2s(self.cloud2_tensor), tf.float32)

		#remove  points closer than minimum radial distance
		not_too_close1 = tf.where(self.cloud1_tensor_spherical[:,0] > self.min_cell_distance)[:,0]
		self.cloud1_tensor_spherical = tf.gather(self.cloud1_tensor_spherical, not_too_close1)
		self.cloud1_tensor = tf.gather(self.cloud1_tensor, not_too_close1)
		not_too_close2 = tf.where(self.cloud2_tensor_spherical[:,0] > self.min_cell_distance)[:,0]
		self.cloud2_tensor_spherical = tf.gather(self.cloud2_tensor_spherical, not_too_close2)
		self.cloud2_tensor = tf.gather(self.cloud2_tensor, not_too_close2)
		
		self.grid_spherical( draw = False )

		# #-----------------------------------------------------------
		# #debug - draw points within a single occupied cell  
		# # working on simple cluster but not on KITTI data
		
		# n = 56 #nth cell with sufficient number of points
		# idx = enough1[n]
		# # print(idx)
		# # print("npts1", npts1[idx])
		# # print("inside1[idx]", inside1[idx])
		# temp = tf.gather(self.cloud1_tensor, inside1[idx]).numpy()
		# # print(temp)
		# self.disp.append(Points(temp, c = 'green', r = 10))
		# # self.draw_cell(tf.gather(o, idx))
		# # print(o)
		# # print(tf.gather(o, idx)[None, None])
		# self.draw_cell(tf.gather(o, idx)[None])

		# # self.draw_ell(mu1, sigma1)
		# # print(tf.shape(mu1))
		# #------------------------------------------------------------

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#draw 2 shells (for debug)
		# z = tf.cast(tf.linspace(0, (self.fid_phi-1)*self.fid_theta*3 - 1, (self.fid_phi-1)*self.fid_theta*3), tf.int32)
		# self.draw_cell(z)
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		x0 = tf.constant([0.1, 0.3, 0., 0., 0., 0.])
		self.main(niter = 5, x0 = x0)

		# self.disp.append(addons.LegendBox(self.disp))
		self.plt.show(self.disp, "Spherical ICET")

	def main(self, niter, x0):
		""" main loop """

		self.X = x0

		self.draw_car()
		o = self.get_occupied()
		# print("occupied = ", o)
		self.draw_cell(o)

		inside1, npts1 = self.get_points_inside(self.cloud1_tensor_spherical,o[:,None])		
		mu1, sigma1 = self.fit_gaussian(self.cloud1_tensor, inside1, tf.cast(npts1, tf.float32))
		enough1 = tf.where(npts1 > self.min_num_pts)[:,0]
		mu1_enough = tf.gather(mu1, enough1)
		sigma1_enough = tf.gather(sigma1, enough1)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		## highlight points in usable cells from scan 1
		# for z in enough1:
		# 	temp = tf.gather(self.cloud1_tensor, inside1[z]).numpy()
		# 	self.disp.append(Points(temp, c = 'green', r = 6))
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# U, L = self.get_U_and_L(sigma1_enough, tf.gather(o, enough1))
		U, L = self.get_U_and_L(sigma1, o)
		# print("U: \n", U)
		# print("L: \n", L)
		self.visualize_L(mu1_enough, U, L)

		self.draw_ell(mu1_enough, sigma1_enough, pc = 1, alpha = 0.5)
		self.draw_cloud(self.cloud1_tensor.numpy(), pc = 1)

		for i in range(niter):
			print("estimated solution vector X: \n", self.X)

			#transform cartesian point cloud 2 by estimated solution vector X
			t = self.X[:3]
			rot = R_tf(-self.X[3:])
			self.cloud2_tensor = tf.matmul((self.cloud2_tensor_OG + t), tf.transpose(rot)) 

			#convert back to spherical coordinates
			self.cloud2_tensor_spherical = tf.cast(self.c2s(self.cloud2_tensor), tf.float32)

			#find points from scan 2 that fall inside occupied voxels
			inside2, npts2 = self.get_points_inside(self.cloud2_tensor_spherical, o[:,None])

			#fit gaussians distributions to each of these groups of points 		
			mu2, sigma2 = self.fit_gaussian(self.cloud2_tensor, inside2, tf.cast(npts2, tf.float32))
			enough2 = tf.where(npts2 > self.min_num_pts)[:,0]
			mu2_enough = tf.gather(mu2, enough2)
			#draw all distribution from scan 2, even in cells without distribution from scan1
			# sigma2_enough = tf.gather(sigma2, enough2)
			# self.draw_ell(mu2_enough, sigma2_enough, pc = 2, alpha = 0.5)

			#get correspondences
			corr = tf.sets.intersection(enough1[None,:], enough2[None,:]).values
			# print("corr indices", corr)
			# self.draw_correspondences(mu1, mu2, corr)

			y0_i = tf.gather(mu1, corr)
			sigma0_i = tf.gather(sigma1, corr)
			npts0_i = tf.gather(npts1, corr)

			y_i = tf.gather(mu2, corr)
			sigma_i = tf.gather(sigma2, corr)
			npts_i = tf.gather(npts2, corr)

			U_i = tf.gather(U, corr)
			L_i = tf.gather(L, corr)

			#get matrix containing partial derivatives for each voxel mean
			H = jacobian_tf(tf.transpose(y_i), self.X[3:]) # shape = [num of corr * 3, 6]
			H = tf.reshape(H, (tf.shape(H)[0]//3,3,6)) # -> need shape [#corr//3, 3, 6]
			# print("H \n", H)

			#construct sensor noise covariance matrix
			R_noise = (tf.transpose(tf.transpose(sigma0_i, [1,2,0]) / tf.cast(npts0_i - 1, tf.float32)) + 
						tf.transpose(tf.transpose(sigma_i, [1,2,0]) / tf.cast(npts_i - 1, tf.float32)) )
			R_noise = L_i @ tf.transpose(U_i, [0,2,1]) @ R_noise @ U_i @ tf.transpose(L_i, [0,2,1])

			#take inverse of R_noise to get our weighting matrix
			W = tf.linalg.pinv(R_noise)

			U_iT = tf.transpose(U_i, [0,2,1])
			LUT = L_i @ U_iT

			# use LUT to remove rows of H corresponding to overly extended directions
			H_z = LUT @ H

			HTWH = tf.math.reduce_sum(tf.matmul(tf.matmul(tf.transpose(H_z, [0,2,1]), W), H_z), axis = 0) #was this (which works)
			HTW = tf.matmul(tf.transpose(H_z, [0,2,1]), W)

			L2, lam, U2 = self.check_condition(HTWH)

			# create alternate corrdinate system to align with axis of scan 1 distributions
			z = tf.squeeze(tf.matmul(LUT, y_i[:,:,None]))
			z0 = tf.squeeze(tf.matmul(LUT, y0_i[:,:,None]))	
			dz = z - z0
			dz = dz[:,:,None] #need to add an extra dimension to dz to get the math to work out

			dx = tf.squeeze(tf.matmul( tf.matmul(tf.linalg.pinv(L2 @ lam @ tf.transpose(U2)) @ L2 @ tf.transpose(U2) , HTW ), dz))
			dx = tf.math.reduce_sum(dx, axis = 0)
			# print("\n dx \n", dx)

			self.X += dx

			#get output covariance matrix
			Q = tf.linalg.pinv(HTWH)
			# print("\n Q \n", Q)

			stds = tf.math.sqrt(tf.abs(Q))
			print("stds: \n", tf.linalg.tensor_diag_part(stds))

		if self.draw == True:
			# self.draw_ell(y_i, sigma_i, pc = 2, alpha = 0.5)
			self.draw_cloud(self.cloud2_tensor.numpy(), pc = 2)
			print("\n L2 \n", L2)
			print("\n lam \n", lam)
			print("\n U2 \n", U2)




	def get_U_and_L(self, sigma1, cells):
		""" 	sigma1 = sigmas from the first scan
				cells = tensor containing the indices of each scan

				U = rotation matrix for each voxel to transform scan 2 distribution
				 into frame of corresponding to ellipsoid axis in keyframe
			    L = matrix to prune extended directions in each voxel (from keyframe)
		
				# starting out by constructing this similar to 3D-ICET, eventually the plan is to try Jason's unscented KF strategy"""

		eigenval, eigenvec = tf.linalg.eig(sigma1)
		U = tf.math.real(eigenvec)

		#need to create [N,3,3] diagonal matrices for axislens
		zeros = tf.zeros([tf.shape(tf.math.real(eigenval))[0]])
		axislen = tf.Variable([tf.math.real(eigenval)[:,0], zeros, zeros,
							   zeros, tf.math.real(eigenval)[:,1], zeros,
							   zeros, zeros, tf.math.real(eigenval)[:,2]])

		axislen = tf.reshape(tf.transpose(axislen), (tf.shape(axislen)[1], 3, 3))
		# print("\n axislen \n", axislen)

		# get projections of axis length in each direction
		rotated = tf.abs(tf.matmul(U,axislen))
		# print("rotated", rotated)

		#need information on the cell index to be able to perform truncation 
		#	-> (cells further from vehicle will require larger distribution length thresholds)
		shell = cells//(self.fid_theta*(self.fid_phi - 1))
		# print("shell", shell)
		r_grid, _ = tf.unique(self.grid[:,0])
		# print("r_grid", r_grid)
		cell_width = tf.experimental.numpy.diff(r_grid)
		# print("cell_width", cell_width)
		# thresholds = (tf.gather(cell_width, shell)**2)/32
		thresholds = (tf.gather(cell_width, shell)**2)/64

		#tile to so that each threshold is repeated 3 times (for each axis)
		thresholds = tf.reshape(tf.transpose(tf.reshape(tf.tile(thresholds[:,None], [3,1]), [3,-1])), [-1,3])[:,None]
		# print("thresholds", thresholds)

		greater_than_thresh = tf.math.greater(rotated, thresholds)
		# print(greater_than_thresh)
		ext_idx = tf.math.reduce_any(greater_than_thresh, axis = 1)
		compact = tf.where(tf.math.reduce_any(tf.reshape(ext_idx, (-1,1)), axis = 1) == False)
		compact =  tf.cast(compact, tf.int32)
		data = tf.ones((tf.shape(compact)[0],3))
		I = tf.tile(tf.eye(3), (tf.shape(U)[0], 1))

		mask = tf.scatter_nd(indices = compact, updates = data, shape = tf.shape(I))

		L = mask * I
		L = tf.reshape(L, (tf.shape(L)[0]//3,3,3))

		return(U,L)

	def visualize_L(self, y0, U, L):
		""" for each voxel center, mu, this func draws untruncated axis via L 
			transformed into the frame of the distribution ellipsoids via U  """

		for i in range(tf.shape(y0)[0]):
			ends =  L[i] @ tf.transpose(U[i])
			arrow_len = 0.5
			arr1 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[0,:]).numpy(), c = 'red')
			self.disp.append(arr1)
			arr2 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[1,:]).numpy(), c = 'green')
			self.disp.append(arr2)
			arr3 = shapes.Arrow(y0[i].numpy(), (y0[i] + arrow_len * ends[2,:]).numpy(), c = 'blue')
			self.disp.append(arr3)

	def check_condition(self, HTWH):
		"""verifies that HTWH is invertable and if not, 
			reduces dimensions to make inversion possible

			L2 = identity matrix which keeps non-extended axis of solution
			lam = diagonal eigenvalue matrix
			U2 = rotation matrix to transform for L2 pruning 
			"""

		cutoff = 1e4 #1e5 #TODO-> experiment with this to get a good value

		#do eigendecomposition
		eigenval, eigenvec = tf.linalg.eig(HTWH)
		eigenval = tf.math.real(eigenval)
		eigenvec = tf.math.real(eigenvec)

		# print("\n eigenvals \n", eigenval)
		# print("\n eigenvec \n", eigenvec)
		# print("\n eigenvec.T \n", tf.transpose(eigenvec))

		#sort eigenvals by size -default sorts small to big
		# small2big = tf.sort(eigenval)
		# print("\n sorted \n", small2big)

		#test if condition number is bigger than cutoff
		condition = eigenval[-1] / eigenval[0]
		# print("\n condition \n", tf.experimental.numpy.log10(abs(condition)))
		# print("\n condition \n", condition.numpy())


		everyaxis = tf.cast(tf.linspace(0,5,6), dtype=tf.int32)
		remainingaxis = everyaxis
		i = tf.Variable([0],dtype = tf.int32) #count var
		#loop until condition number is small enough to make matrix invertable
		while abs(condition) > cutoff:

			condition = eigenval[-1] / tf.gather(eigenval, i)
			# print("condition", tf.experimental.numpy.log10(abs(condition)).numpy())

			if abs(condition) > cutoff:
				i.assign_add(tf.Variable([1],dtype = tf.int32))
				remainingaxis = everyaxis[i.numpy()[0]:]

		#create identity matrix truncated to only have the remaining axis
		L2 = tf.gather(tf.eye(6), remainingaxis)

		# #alternate strategy- zero out instead of keeping axis truncated
		# while tf.shape(L2)[0] < 6:
		# 	L2 = tf.concat((tf.zeros([1,6]), L2), axis = 0)

		# print("\n L2 \n", L2)

		U2 = eigenvec
		# print("\n U2 \n", U2)

		lam = tf.eye(6)*eigenval
		# print("\n lam \n", lam)

		return(L2, lam, U2)


	def draw_ell(self, mu, sigma, pc = 1, alpha = 1):
		"""draw distribution ellipses given mu and sigma tensors"""

		if pc == 1:
			color = [0.8, 0.3, 0.3]
		if pc ==2:
			color = [0.3, 0.3, 0.8]

		for i in range(tf.shape(sigma)[0]):

			eig = np.linalg.eig(sigma[i,:,:].numpy())
			eigenval = eig[0] #correspond to lengths of axis
			eigenvec = eig[1]

			# assmues decreasing size
			a1 = eigenval[0]
			a2 = eigenval[1]
			a3 = eigenval[2]

			if mu[i,0] != 0 and mu[i,1] != 0:
				ell = Ell(pos=(mu[i,0], mu[i,1], mu[i,2]), axis1 = 4*np.sqrt(abs(a1)), 
					axis2 = 4*np.sqrt(abs(a2)), axis3 = 4*np.sqrt(abs(a3)), 
					angs = (np.array([-R2Euler(eigenvec)[0], -R2Euler(eigenvec)[1], -R2Euler(eigenvec)[2] ])), c=color, alpha=alpha, res=12)
				
				self.disp.append(ell)

	def draw_correspondences(self, mu1, mu2, corr):
		""" draw arrow between distributions between scans that:
			1- contain sufficient number of points 
			2- occupy the same voxel """

		# print("correspondences", corr)
		for i in corr:
			a = shapes.Arrow(mu2[i].numpy(), mu1[i].numpy(), c = "black")
			self.disp.append(a)

	def get_corners(self, cells):
		""" returns  spherical coordinates of coners of each input cell 
			cells = tensor containing cell indices """

		#to account for wrapping around at end of each ring
		per_shell = self.fid_theta*(self.fid_phi - 1) #number of cells per radial shell
		fix =  (self.fid_phi*self.fid_theta)*((((cells)%per_shell) + (self.fid_phi-1) )//per_shell)
		n = cells + cells//(self.fid_phi - 1)

		p1 = tf.gather(self.grid, n)
		p2 = tf.gather(self.grid, n+self.fid_phi - fix)
		p3 = tf.gather(self.grid, n + self.fid_theta*self.fid_phi)
		p4 = tf.gather(self.grid, n + self.fid_phi + (self.fid_theta*self.fid_phi) - fix)
		p5 = tf.gather(self.grid, n + 1)
		p6 = tf.gather(self.grid, n+self.fid_phi +1 - fix)
		p7 = tf.gather(self.grid, n + (self.fid_theta*self.fid_phi) + 1)
		p8 = tf.gather(self.grid, n + self.fid_phi + (self.fid_theta*self.fid_phi) +1 - fix)

		out = tf.transpose(tf.Variable([p1, p2, p3, p4, p5, p6, p7, p8]), [1, 0, 2])

		return(out)


	def fit_gaussian(self, cloud, rag, npts):
		""" fits 3D gaussian distribution to each elelment of 
			rag, which cointains indices of points in cloud """
		st = time.time()

		coords = tf.gather(cloud, rag)
		mu = tf.math.reduce_mean(coords, axis=1)
		#for debug ------
		# self.disp.append(Points( mu.numpy(), c = 'b', r = 20 ))
		#----------------

		xpos = tf.gather(cloud[:,0], rag)
		ypos = tf.gather(cloud[:,1], rag)
		zpos = tf.gather(cloud[:,2], rag)
		# print(xpos)
		# print("mux",mu[:,0])

		xx = tf.math.reduce_sum(tf.math.square(xpos - mu[:,0][:,None] ), axis = 1)/npts
		yy = tf.math.reduce_sum(tf.math.square(ypos - mu[:,1][:,None] ), axis = 1)/npts
		zz = tf.math.reduce_sum(tf.math.square(zpos - mu[:,2][:,None] ), axis = 1)/npts

		xy = tf.math.reduce_sum( (xpos - mu[:,0][:,None])*(ypos - mu[:,1][:,None]), axis = 1)/npts
		xz = tf.math.reduce_sum( (xpos - mu[:,0][:,None])*(zpos - mu[:,2][:,None]), axis = 1)/npts
		yz = tf.math.reduce_sum( (ypos - mu[:,1][:,None])*(zpos - mu[:,2][:,None]), axis = 1)/npts


		sigma = tf.Variable([xx, xy, xz,
							 xy, yy, yz,
							 xz, yz, zz]) 
		sigma = tf.reshape(tf.transpose(sigma), (tf.shape(sigma)[1] ,3,3))
		# print("sigma", sigma)

		# print("took", time.time()-st, "s to fit gaussians to each cell")
		return(mu, sigma)


	def get_points_inside(self, cloud, cells):
		""" returns ragged tensor containing the indices of points in <cloud> inside each cell in <cells>"""
		st = time.time()
		# print("specified cells:", cells)

		thetamin = -np.pi
		thetamax = np.pi #-  2*np.pi/self.fid_theta
		phimin =  3*np.pi/8
		phimax = 5*np.pi/8 
		# print(self.grid)

		edges_r, _ = tf.unique(self.grid[:,0])
		bins_r = tfp.stats.find_bins(cloud[:,0], edges_r)
		
		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)
		# edges_theta, _ = tf.unique(self.grid[:,1])
		bins_theta = tfp.stats.find_bins(cloud[:,1], edges_theta)
		# print("edges_theta", edges_theta)
		# print("bins_theta", bins_theta)

		edges_phi = tf.linspace(phimin, phimax, self.fid_phi)
		# edges_phi, _ = tf.unique(self.grid[:,2])
		bins_phi = tfp.stats.find_bins(cloud[:,2], edges_phi)		

		#cell index for every point in cloud
		cell_idx = tf.cast( bins_theta*(self.fid_phi-1) + bins_phi + bins_r*self.fid_theta*(self.fid_phi-1), tf.int32) #was this pre 3/5 (incorrect??)



		# print("cell index for each point", cell_idx)

		pts_in_c = tf.where(cell_idx == cells)
		# print("cell ID for each point", pts_in_c[:,0])

		#TODO: check for skipped indices here -> it may be producing a bug
		# print(tf.unique(pts_in_c[:,0]))
		
		numPtsPerCell = tf.math.bincount(tf.cast(pts_in_c[:,0], tf.int32))
		# print("numPtsPerCell", numPtsPerCell)
		#test
		# _, num_pts = tf.unique(pts_in_c[:,0])
		# print("num_pts", num_pts)

		pts_in_c = tf.RaggedTensor.from_value_rowids(pts_in_c[:,1], pts_in_c[:,0]) 
		# pts_in_c = tf.RaggedTensor.from_value_rowids(pts_in_c[:,1], pts_in_c[:,0], nrows = tf.shape(cells)[0].numpy())
		# print(pts_in_c.get_shape())
		# print(tf.shape(cells))
		# print("pts_in_c", pts_in_c)

		# print("index of points in specified cell", pts_in_c)

		# print("took", time.time()-st, "s to find pts in cells")
		return(pts_in_c, numPtsPerCell)


	def get_occupied(self):
		""" returns idx of all voxels that occupy the line of sight closest to the observer """

		st = time.time()

		#attempt #2:------------------------------------------------------------------------------
		#bin points by spike
		thetamin = -np.pi
		thetamax = np.pi #-  2*np.pi/self.fid_theta
		phimin =  3*np.pi/8
		phimax = 5*np.pi/8 

		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta + 1)
		bins_theta = tfp.stats.find_bins(self.cloud1_tensor_spherical[:,1], edges_theta)
		# print(bins_theta)
		edges_phi = tf.linspace(phimin, phimax, self.fid_phi)
		bins_phi = tfp.stats.find_bins(self.cloud1_tensor_spherical[:,2], edges_phi)
		# print(bins_phi)

		#combine bins_theta and bins_phi to get spike bins
		bins_spike = tf.cast(bins_theta*(self.fid_phi-1) + bins_phi, tf.int32)
		# print(tf.unique(bins_spike))
		# print("bins_spike", bins_spike)
		# self.draw_cell(tf.cast(bins_spike, tf.int32))

		#find min point in each occupied spike
		occupied_spikes, idxs = tf.unique(bins_spike)
		# print("occupied_spikes:", occupied_spikes)
		# print(idxs)

		temp =  tf.where(bins_spike == occupied_spikes[:,None]) #TODO- there has to be a better way to do this... 
		# print(temp)
		rag = tf.RaggedTensor.from_value_rowids(temp[:,1], temp[:,0])
		# print(rag)

		idx_by_rag = tf.gather(self.cloud1_tensor_spherical[:,0], rag)
		# print(idx_by_rag)

		min_per_spike = tf.math.reduce_min(idx_by_rag, axis = 1)
		# print("min_per_spike:", min_per_spike)

		#get closest shell for each point in min_per_spike
		# print(tf.unique(self.grid[:,0]))
		radii, _ = tf.unique(self.grid[:,0])
		# print(radii)
		shell_idx = tf.math.reduce_sum(tf.cast(tf.greater(min_per_spike, radii[:,None] ), tf.int32), axis = 0) - 1
		# print(shell_idx)

		#find bin corresponding to the identified closeset points per cell
		occupied_cells = occupied_spikes + shell_idx*self.fid_theta*(self.fid_phi -1)
		# print("occupied_cells:", occupied_cells)

		# print("took", time.time() - st, "s to find occupied cells")
		return(occupied_cells)
		#-----------------------------------------------------------------------------------------



		# #attempt #1: directly find which bins points are in (inefficient)-----------------------

		# #get spherical coordinates of all of the corners of the cells in the innermost shell
		# # shell_cells = tf.constant([1,300]) #debug
		# shell_cells = tf.cast(tf.linspace(0, (self.fid_theta-1)*(self.fid_phi - 1) - 1, tf.cast((self.fid_theta-1)*(self.fid_phi - 1), tf.int32)), tf.int32)

		# corn = self.get_corners(shell_cells)
		# # print(corn)
		# rmin = corn[:,0,0][:,None]
		# thetamin = corn[:,0,1][:,None]
		# thetamax = corn[:,3,1][:,None]
		# phimin = corn[:,0,2][:,None]
		# phimax = corn[:,4,2][:,None]

		# # #for debug: draw all corners of cells in shell ----------------
		# # for i in range(tf.shape(shell_cells)[0]):
		# # 	pts = self.s2c(corn[i])
		# # 	self.disp.append(Points(pts.numpy(), c = 'red', r =10))
		# # #--------------------------------------------------------------

		# inside = tf.Variable([tf.greater(self.cloud1_tensor_spherical[:,0], rmin),
		# 					tf.less(self.cloud1_tensor_spherical[:,1], thetamax),
		# 					tf.greater(self.cloud1_tensor_spherical[:,1], thetamin)
		# 					])
		# inside = tf.transpose(inside, [1,2,0])
		# # print(inside)

		# combined = tf.math.reduce_all(inside, axis = 2)
		# # print(combined)
		# #--------------------------------------------------------------------------------------


	def draw_cell(self, idx):
		""" draws cell provided by idx tensor"""

		corners = self.get_corners(idx)
		# print(corners)

		for i in range(tf.shape(corners)[0]):

			p1, p2, p3, p4, p5, p6, p7, p8 = self.s2c(corners[i]).numpy()
			arc1 = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'red')	
			# arc1 = shapes.Line(p1, p2, c = 'red', lw = 1) #debug		
			self.disp.append(arc1)
			arc2 = shapes.Arc(center = [0,0,0], point1 = p3, point2 = p4, c = 'red')
			# arc2 = shapes.Line(p3, p4, c = 'red', lw = 1) #debug
			self.disp.append(arc2)
			line1 = shapes.Line(p1, p3, c = 'red', lw = 1)
			self.disp.append(line1)
			line2 = shapes.Line(p2, p4, c = 'red', lw = 1) #problem here
			self.disp.append(line2)

			arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'red')			
			self.disp.append(arc3)
			arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'red')
			self.disp.append(arc4)
			line3 = shapes.Line(p5, p7, c = 'red', lw = 1)
			self.disp.append(line3)
			line4 = shapes.Line(p6, p8, c = 'red', lw = 1)
			self.disp.append(line4)

			self.disp.append(shapes.Line(p1,p5,c = 'red', lw = 1))
			self.disp.append(shapes.Line(p2,p6,c = 'red', lw = 1))
			self.disp.append(shapes.Line(p3,p7,c = 'red', lw = 1))
			self.disp.append(shapes.Line(p4,p8,c = 'red', lw = 1))

	def grid_spherical(self, draw = False):
		""" constructs grid in spherical coordinates """

		self.fid_r = self.fid  #num radial division
		self.fid_theta = self.fid  #number of subdivisions in horizontal directin
		self.fid_phi = self.fid_theta//6 #number of subdivision in vertical direction + 1

		thetamin = -np.pi 
		thetamax = np.pi - 2*np.pi/self.fid_theta
		phimin =  3*np.pi/8
		phimax = 5*np.pi/8 

		a = tf.cast(tf.linspace(0,self.fid_r-1, self.fid_r)[:,None], tf.float32)
		b = tf.linspace(thetamin, thetamax, self.fid_theta)[:,None]
		c = tf.linspace(phimin, phimax, self.fid_phi)[:,None]

		ansb = tf.tile(tf.reshape(tf.tile(b, [1,self.fid_phi]), [-1,1] ), [(self.fid_r), 1])
		ansc = tf.tile(c, [self.fid_theta*self.fid_r, 1])
		#need to iteratively adjust spacing of radial positions to make cells roughly cubic

		nshell = self.fid_theta*(self.fid_phi) #number of grid cells per shell
		r_last = self.min_cell_distance #radis of line from observer to previous shell
		temp = np.ones([tf.shape(ansc)[0], 1])*self.min_cell_distance
		for i in range(1,self.fid_r):
			r_new = r_last*(1 + (np.arctan(2*np.pi/self.fid_theta)))
			temp[(i*nshell):((i+1)*nshell+1),0] = r_new
			r_last = r_new
		ansa = tf.convert_to_tensor(temp, tf.float32)

		self.grid = tf.cast(tf.squeeze(tf.transpose(tf.Variable([ansa,ansb,ansc]))), tf.float32)
		# print(self.grid)

		if draw == True:
			gp = self.s2c(self.grid.numpy())
			# print(gp)
			p = Points(gp, c = [0.3,0.8,0.3], r = 5)
			self.disp.append(p)

	def c2s(self, pts):
		""" converts points from cartesian coordinates to spherical coordinates """
		r = tf.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
		phi = tf.math.acos(pts[:,2]/r)
		theta = tf.math.atan2(pts[:,1], pts[:,0])

		out = tf.transpose(tf.Variable([r, theta, phi]))
		return(out)

	def s2c(self, pts):
		"""converts spherical -> cartesian"""

		x = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.cos(pts[:,1])
		y = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.sin(pts[:,1]) 
		z = pts[:,0]*tf.math.cos(pts[:,2])

		out = tf.transpose(tf.Variable([x, y, z]))
		# out = tf.Variable([x, y, z])
		return(out)

	def draw_cloud(self, points, pc = 1):

		if pc == 1:
			color = [0.8, 0.5, 0.5]
		if pc == 2:
			color = [0.5, 0.5, 0.8]
		
		c = Points(points, c = color, r = 4)
		self.disp.append(c)

	def draw_car(self):
		# (used for making presentation graphics)
		fname = "C:/Users/Derm/honda.vtk"
		car = Mesh(fname).c("gray").rotate(90, axis = (0,0,1)).addShadow(z=-1.85)
		car.pos(1.4,1,-1.72)
		# car.orientation(vector(0,np.pi/2,0)) 
		self.disp.append(car)
		#draw red sphere at location of sensor
		self.disp.append(Points(np.array([[0,0,0]]), c = [0.9,0.9,0.5], r = 10))

		# print(car.rot)