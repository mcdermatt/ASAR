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

		self.cloud1_tensor_spherical = self.c2s(self.cloud1_tensor)
		self.grid_spherical( draw = False )



		self.draw_cloud(cloud1)
		self.draw_car()
		self.plt.show(self.disp, "Spherical ICET")


	def get_occupied(self, ):
		""" returns idx of all voxels that occupy the line of sight closest to the observer """

		# corners = self.get_corners()
		


	def grid_spherical(self, draw = False):
		""" constructs grid in spherical coordinates """

		self.fid_r = self.fid  #num radial division
		self.fid_theta = self.fid  #number of subdivisions in horizontal directin
		self.fid_phi = self.fid_theta//6 #number of subdivision in vertical direction + 1

		thetamin = -np.pi + 2*np.pi/self.fid_theta
		thetamax = np.pi
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