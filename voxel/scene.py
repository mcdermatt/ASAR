import numpy as np
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget

class scene():

	""" cubic voxel grid used for occlusion culling demo """


	def __init__(self, cloud = None, fid = 20):

		self.fid = fid # dimension of 3D grid: [fid, fid, fid]
		self.cloud = cloud
		self.mnp = 1 #minimum number of points to count as occupied
		self.numvox = (self.fid+1)*(self.fid + 1)*(self.fid + 1) #number of voxels
		self.wire = True #draw cells as wireframe

		self.plt = Plotter(N = 1, axes = 4, bg = (1, 1, 1), interactive = True)
		self.disp = []

		self.occupancy_grid(draw = False)
		# self.cull()

		self.draw_cloud()
		self.draw_car()
		self.plt.show(self.disp, "3D Occlusion Culling Test")


	def cull(self):
		""" culls all voxels occluded from POV of observer """
		
		#loop through each occupied voxel
		for o in self.occupied:

			#get outside corners of cell 
			voxctr = np.squeeze(self.grid[o])
			corners = np.ones([8,3]) * voxctr
			corners[:,0] -= self.cw/2			
			corners[2:4,0] += self.cw
			corners[6:,0] += self.cw
			corners[:,1] -= self.cw/2			
			corners[0:-1:2,1] += self.cw
			corners[:4,2] += self.cw/2
			corners[4:,2] -= self.cw/2
			# print(corners)
			# print(voxctr)

			#get angle to center of cell
			#rotation about vertical axis
			# theta = np.sin(voxctr[0]/voxctr[1]) #wrong?
			theta = np.arctan2(voxctr[1],voxctr[0])
			if np.isnan(theta) == True:
				theata = 0
			# print("theta", theta)
			#rotation up/down
			# phi = 
			dthetamax = 0
			dthetamin = 0
			for i in range(8):			#loop through each corner
				#find corners with largest horizonatal rotation band
				# theta_i = np.sin(corners[i,0]/corners[i,1])
				theta_i = np.arctan2(corners[i,1],corners[i,0])
				# print("theta_i", theta_i)
				dtheta = theta_i - theta
				# print("dtheta", dtheta)
				if dtheta > dthetamax:
					dthetamax = dtheta
					thetamax = i
				if dtheta < dthetamin:
					dthetamin = dtheta
					thetamin = i				

				#find corners with largest vertical rotation band			
					#...

			L_theta_min = shapes.Line(np.array([0,0,0]), corners[thetamin], lw = 4, c ='b')
			L_theta_max = shapes.Line(np.array([0,0,0]), corners[thetamax], lw = 4, c='b')
			self.disp.append(L_theta_min)
			self.disp.append(L_theta_max)

			corn = Points(corners, c = [0.5,0.9,0.5], r = 10)
			self.disp.append(corn)

			#find grid centers that fall between thetamin and thetamax
			grid_thetas = np.arctan2(self.grid[:,1], self.grid[:,0])
			between = np.asarray(np.where(np.array([ grid_thetas < thetamax,
										  grid_thetas > thetamin
											]).all(axis = 0) == True)).T
			# print(between)
			for b in between:
				self.draw_cell(b, wire = False)

	def occupancy_grid(self, draw = True):

		""" constructs occupancy grid from input point cloud """
		minxy = -50 #-100
		maxxy = 50 #100
		minz = -50#-2
		maxz = 50#3#8

		self.cw = 100/self.fid #cell width

		self.grid = np.mgrid[minxy:maxxy:(self.fid+1)*1j, minxy:maxxy:(self.fid+1)*1j, minz:maxz:(self.fid + 1)*1j]
		self.grid = np.flip(np.reshape(self.grid, (3,-1), order = 'C').T, axis = 0)
		if draw:
			g = Points(self.grid, c = [0.8, 0.5, 0.5], r = 4)
			self.disp.append(g)


		has_pts = np.zeros(self.numvox)

		#loop through all voxels
		for j in range(self.numvox):

			# if j%1000 == 0:
			# 	print(j)

			#test if there are points in lidar scan in voxel j
			inside =  np.where(np.array([self.cloud[:,0] > self.grid[j,0] - self.cw/2,  # greater than minx 
							  			self.cloud[:,0] < self.grid[j,0] + self.cw/2,  # less than maxx
										self.cloud[:,1] > self.grid[j,1] - self.cw/2,
										self.cloud[:,1] < self.grid[j,1] + self.cw/2,
										self.cloud[:,2] > self.grid[j,2] - self.cw/2,
										self.cloud[:,2] < self.grid[j,2] + self.cw/2,
										]).all(axis = 0) == True)

			if np.shape(inside)[1] >= self.mnp:
				self.draw_cell(j)
				has_pts[j] = 1

		self.occupied = np.asarray(np.where(has_pts == 1)).T

	def draw_cell(self, cell, wire = True):
		"""draw specified cell number"""
	
		#get defining boundary of cell

		#for convex hull--------
		# pts = self.grid[np.array([0, 10, 151, 23, 55])]
		# h = shapes.ConvexHull(pts).c("red").alpha(1)
		# self.disp.append(h)
		#-----------------------

		#for simple cube--------
		# s = [xmin,xmax, ymin,ymax, zmin,zmax]
		s = [self.grid[cell,0] - self.cw/2, self.grid[cell,0] + self.cw/2, 
			 self.grid[cell,1] - self.cw/2, self.grid[cell,1] + self.cw/2,
			 self.grid[cell,2] - self.cw/2, self.grid[cell,2] + self.cw/2 ]
		b = shapes.Box(size = s, c = [0.2,0.2,0.5]).wireframe(wire)
		self.disp.append(b)


	def draw_cloud(self):
		c = Points(self.cloud, c = [0.5, 0.5, 0.8], r = 4)
		self.disp.append(c)
		
	def draw_car(self):
		# (used for making presentation graphics)
		fname = "C:/Users/Derm/honda.vtk"
		car = Mesh(fname).c("gray").addShadow(z=-1.85)
		car.pos(1.,-1,-1.72) 
		self.disp.append(car)
		#draw red sphere at location of sensor
		self.disp.append(Points(np.array([[0,0,0]]), c = [0.9,0.5,0.5], r = 10))