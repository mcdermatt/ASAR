import numpy as np
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget

class scene():

	""" cubic voxel grid used for occlusion culling demo """


	def __init__(self, cloud = None, fid = 20, cull = False, coord = 0):
		""" coord: 0 == cartesian, 1 == spherical"""

		self.fid = fid # dimension of 3D grid: [fid, fid, fid]
		self.cloud = cloud
		self.mnp = 1 #minimum number of points to count as occupied
		self.wire = True #draw cells as wireframe
		self.coord = coord
		self.plt = Plotter(N = 1, axes = 4, bg = (1, 1, 1), interactive = True)
		self.disp = []

		self.numvox = (self.fid+1)*(self.fid + 1)*(self.fid//10 + 1) #number of voxels

		if self.coord == 0:
			self.occupancy_grid_cart(draw = False)
			if cull == True:
				self.cull()
				for b in np.squeeze(np.asarray(np.where(self.occluded == 1))):
					self.draw_cell(int(b), wire = False)

		if self.coord == 1:
			self.occupancy_grid_spherical()
			cnum = 12
			print(self.grid[cnum])
			self.draw_cell(cnum)

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
			#rotation up/down
			phi = np.arctan2(voxctr[2], np.sqrt(voxctr[0]**2 + voxctr[1]**2))
			if np.isnan(phi) == True:
				phi = 0

			dthetamax = 0
			dthetamin = 0
			dphimax = 0
			dphimin = 0
			for i in range(8):			#loop through each corner
				#find corners with largest horizonatal rotation band
				theta_i = np.arctan2(corners[i,1],corners[i,0])
				dtheta = theta_i - theta
				if dtheta > dthetamax:
					dthetamax = dtheta
					thetamax = theta_i
					maxcorner = i
				if dtheta < dthetamin:
					dthetamin = dtheta
					thetamin = theta_i
					mincorner = i	

				#find corners with largest vertical rotation band			
				phi_i = np.arctan2(corners[i, 2], np.sqrt(corners[i,0]**2 + corners[i,1]**2))
				dphi = phi_i - phi
				if dphi > dphimax:
					dphimax = dphi
					phimax = phi_i
					#todo: draw corners for phi...
				if dphi < dphimin:
					dphimin = dphi
					phimin = phi_i

			# #for debug ~~~~~~~~~~~~~~~~~~~~~~~~
			# #draw lines to outisde corners
			# L_theta_min = shapes.Line(np.array([0,0,0]), corners[mincorner], lw = 4, c ='b')
			# L_theta_max = shapes.Line(np.array([0,0,0]), corners[maxcorner], lw = 4, c='b')
			# self.disp.append(L_theta_min)
			# self.disp.append(L_theta_max)
			# #highlight corners
			# corn = Points(corners, c = [0.5,0.9,0.5], r = 10)
			# self.disp.append(corn)
			# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

			#find grid centers that fall between thetamin and thetamax
			grid_thetas = np.arctan2(self.grid[:,1], self.grid[:,0])
			grid_phis = np.arctan2(self.grid[:, 2], np.sqrt(self.grid[:,0]**2 + self.grid[:,1]**2))
			# print(np.shape(grid_thetas))

			d2o = np.sqrt(np.sum(voxctr**2)) #distance from oberver to center of o 
			# print(d2o)
			d2vc = np.sqrt(np.sum(self.grid[:]**2, axis = 1)) #distance to each voxel center
			# print(d2vc)

			blocked = np.asarray(np.where(np.array([ grid_thetas < thetamax, #must fall within frusturm of occlusion 
										  			 grid_thetas > thetamin, #horizontal compoent
										  			 grid_phis < phimax, #vertical component
										  			 grid_phis > phimin, #vertical component
													 d2vc > d2o#,		  #must be further away from the observer than the center of the occluding voxel
													 # d2vc > 3, 	#must be some minimum distance away (to ignore artifacts from KITTI ego vehicle)
											]).all(axis = 0) == True)).T
			blocked = np.squeeze(blocked)
			self.occluded[blocked] = 1

	def occupancy_grid_cart(self, draw = True):
		""" constructs cartesian occupancy grid from input point cloud """
		minxy = -50 #-100
		maxxy = 50 #100
		minz = -3
		maxz = 7

		self.cw = 100/self.fid #cell width
		self.grid = np.mgrid[minxy:maxxy:(self.fid+1)*1j, minxy:maxxy:(self.fid+1)*1j, minz:maxz:(self.fid//10 + 1)*1j]
		self.grid = np.flip(np.reshape(self.grid, (3,-1), order = 'C').T, axis = 0)
		#init arr for storing info on if a voxel is occluded
		self.occluded = np.zeros(np.shape(self.grid)[0])

		if draw:
			g = Points(self.grid, c = [0.8, 0.5, 0.5], r = 4)
			self.disp.append(g)
		has_pts = np.zeros(self.numvox)

		for j in range(self.numvox): #loop through all voxels
			#test if there are points in lidar scan in voxel j
			inside =  np.where(np.array([self.cloud[:,0] > self.grid[j,0] - self.cw/2,  # greater than minx 
							  			self.cloud[:,0] < self.grid[j,0] + self.cw/2,  # less than maxx
										self.cloud[:,1] > self.grid[j,1] - self.cw/2,
										self.cloud[:,1] < self.grid[j,1] + self.cw/2,
										self.cloud[:,2] > self.grid[j,2] - self.cw/2,
										self.cloud[:,2] < self.grid[j,2] + self.cw/2
										]).all(axis = 0) == True)

			if np.shape(inside)[1] >= self.mnp:
				 #don't count stuff too close to ego-vehicle as occupied
				if np.sqrt(np.sum(self.grid[j]**2)) > 7:
					self.draw_cell(j, wire = True)
					has_pts[j] = 1

		self.occupied = np.asarray(np.where(has_pts == 1)).T

	def occupancy_grid_spherical(self, draw = True):
		""" constructs occupancy grid in spherical coordinates """

		fid_r = 3 #self.fid #num radial division
		rmax = 50
		fid_theta = 3 #number of subdivisions in horizontal directin
		thetamin = np.pi/2 #-np.pi / 6
		thetamax = np.pi/4 #np.pi / 6
		fid_phi = 3 #self.fid #number of subdivision in vertical direction
		phimin = np.pi / 4
		phimax = np.pi/ 2

		#establish grid array which describes the center point of each cell
		self.grid = np.mgrid[0:rmax:(fid_r)*1j, thetamin:thetamax:(fid_theta)*1j, phimin:phimax:(fid_phi)*1j]
		self.grid = np.reshape(self.grid, (3,-1), order = 'C').T

		p = Points(self.s2c(self.grid), c = [0.3,0.8,0.3], r = 5)
		self.disp.append(p)


	def s2c(self, arr):
		"""convert spherical coordiances to cartesian"""

		if len(np.shape(arr)) == 1:
			r = arr[0]
			theta = arr[1]
			phi = arr[2]
		else:
			r = arr[:,0]
			theta = arr[:,1]
			phi = arr[:,2]

		x = r*np.sin(phi)*np.cos(theta)
		y = r*np.sin(phi)*np.sin(theta)
		z = r*np.cos(phi)

		return(np.array([x, y, z]))

	def draw_cell(self, cell, wire = False):
		"""draw specified cell number"""
		
		if self.coord == 0:
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

		#for spherical coordinates
		if self.coord == 1:

			#draw arc on side closet to ego-vehicle
			p1 = self.s2c(self.grid[cell])
			p2 = self.s2c(self.grid[cell+3])

			print("p1", p1, "\n p2", p2)

			a = shapes.Arc(center = [0,0,0], point1 = p1, point2 = p2, c = 'red')			
			self.disp.append(a)


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