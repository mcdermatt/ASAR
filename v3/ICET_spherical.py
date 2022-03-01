import numpy as np
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget
import time
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp

class ICET():


	def __init__(self, cloud1, cloud2 = None, fid = 30):

		self.fid = fid # dimension of 3D grid: [fid, fid, fid]
		self.cloud1 = cloud1

		#convert cloud1 to tesnsor
		self.cloud1_tensor = tf.convert_to_tensor(cloud1)
		# print(tf.shape(self.cloud1_tensor))

		# self.mnp = 10 #minimum number of points to count as occupied
		# self.wire = True #draw cells as wireframe
		self.plt = Plotter(N = 1, axes = 4, bg = (1, 1, 1), interactive = True) #axis = 4
		self.disp = []
		self.min_cell_distance = 3 #begin closest spherical voxel here

		self.cloud1_tensor_spherical = tf.cast(self.c2s(self.cloud1_tensor), tf.float32)
		self.grid_spherical( draw = False )

		# test = tf.constant([3, 4, 110])
		test = tf.cast(tf.linspace(0, (self.fid_theta)*(self.fid_phi-1) - 1,(self.fid_theta)*(self.fid_phi-1)), tf.int32)
		# test = tf.cast(tf.linspace(0, 10, 11), tf.int32)
		# print(test)
		self.draw_cell(test)

		self.get_occupied()

		self.draw_cloud(cloud1)
		self.draw_car()
		self.plt.show(self.disp, "Spherical ICET")


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


	def get_occupied(self):
		""" returns idx of all voxels that occupy the line of sight closest to the observer """

		#attempt #2:------------------------------------------------------------------------------
		#bin points by spike
		thetamin = -np.pi + 2*np.pi/self.fid_theta
		thetamax = np.pi
		phimin =  3*np.pi/8
		phimax = 5*np.pi/8 

		edges_theta = tf.linspace(thetamin, thetamax, self.fid_theta)
		bins_theta = tfp.stats.find_bins(self.cloud1_tensor_spherical[:,1], edges_theta)
		# print(bins_theta)
		edges_phi = tf.linspace(phimin, phimax, self.fid_phi)
		bins_phi = tfp.stats.find_bins(self.cloud1_tensor_spherical[:,2], edges_phi)
		# print(bins_phi)

		#combine bins_theta and bins_phi to get spike bins
		bins_spike = bins_theta*(self.fid_phi-1) + bins_phi
		# print(tf.unique(bins_spike))
		print(bins_spike)

		#find min point in each occupied spike

		#find bin corresponding to the identified closeset points per cell



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

	def draw_cloud(self, points):
		c = Points(points, c = [0.5, 0.5, 0.8], r = 4)
		self.disp.append(c)

	def draw_car(self):
		# (used for making presentation graphics)
		fname = "C:/Users/Derm/honda.vtk"
		car = Mesh(fname).c("gray").rotate(90, axis = (0,0,1)).addShadow(z=-1.85)
		car.pos(1.4,1,-1.72)
		# car.orientation(vector(0,np.pi/2,0)) 
		self.disp.append(car)
		#draw red sphere at location of sensor
		self.disp.append(Points(np.array([[0,0,0]]), c = [0.9,0.5,0.5], r = 10))

		# print(car.rot)