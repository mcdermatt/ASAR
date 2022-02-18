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

		self.plt = Plotter(N = 1, axes = 0, bg = (1, 1, 1), interactive = True)
		self.disp = []

		self.occupancy_grid(draw = False)

		self.draw_cloud()
		self.draw_car()
		self.plt.show(self.disp, "3D Occlusion Culling Test")


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

		self.occupied = np.where(has_pts == 1)

	def draw_cell(self, cell):
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
		b = shapes.Box(size = s, c = [0.2,0.2,0.5]).wireframe(self.wire)
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