import numpy as np
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget
import time

class scene():

	""" cubic voxel grid used for occlusion culling demo """


	def __init__(self, cloud = None, fid = 20, cull = False, coord = 0):
		""" coord: 0 == cartesian, 1 == spherical"""

		self.fid = fid # dimension of 3D grid: [fid, fid, fid]
		self.cloud = cloud
		self.mnp = 10 #minimum number of points to count as occupied
		self.wire = True #draw cells as wireframe
		self.coord = coord
		self.plt = Plotter(N = 1, axes = 4, bg = (1, 1, 1), interactive = True) #axis = 4
		self.disp = []
		self.min_cell_distance = 3 #begin closest spherical voxel here

		if self.coord == 0:
			self.numvox = (self.fid+1)*(self.fid + 1)*(self.fid//10 + 1) #number of voxels
			self.occupancy_grid_cart(draw = False)
			if cull == True:
				self.cull()
				for b in np.squeeze(np.asarray(np.where(self.occluded == 1))):
					self.draw_cell(int(b), wire = False)

		if self.coord == 1:
			self.cloud_spherical = self.c2s(self.cloud)
			self.grid_spherical(draw = False)
			# print(self.grid)
			# print(np.shape(self.grid))
			self.get_occupied_spherical()
			for o in self.occupied:
				self.draw_cell(o)
				inside = self.find_pts_in_cell(o)
				self.disp.append(Points(self.cloud[inside], c = [0.4,0.8,0.3], r = 5))

			# for i in range(3*self.fid_theta*(self.fid_phi-1)):
			# 	self.draw_cell(i)

			# self.draw_cell(int(2*self.fid_theta*(self.fid_phi-1) - 1))

		self.draw_cloud()
		self.draw_car()
		self.plt.show(self.disp, "Spherical Grid Test")

		# ViewInteractiveWidget()


	def cull(self):
		""" culls all voxels occluded from POV of observer (for use with cartesian voxels) """
		
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

	def grid_spherical(self, draw = True):
		""" constructs grid in spherical coordinates """

		self.fid_r = self.fid  #num radial division
		self.fid_theta = self.fid  #number of subdivisions in horizontal directin
		self.fid_phi = self.fid_theta//6 #number of subdivision in vertical direction + 1

		# rmax = 100
		# thetamin = -np.pi + 2*np.pi/self.fid_theta
		# thetamax = np.pi
		thetamin = -np.pi
		thetamax = np.pi - 2*np.pi/self.fid_theta 

		phimin =  3*np.pi/8 # np.pi / 4
		phimax = 5*np.pi/8 #np.pi/2 

		#establish grid array which describes the ego front right point of each cell
		#using constant raidus incraments (old) ----------------------
		# self.grid = np.mgrid[0:rmax:(self.fid_r)*1j, thetamin:thetamax:(self.fid_theta)*1j, phimin:phimax:(self.fid_phi)*1j]
		# self.grid = np.reshape(self.grid, (3,-1), order = 'C').T
		# self.grid[:,0] += 3 #start some distance from vehicle
		#-------------------------------------------------------------

		#using increasing radius steps to keep voxels roughly cubic (new) -----
		self.grid = np.mgrid[0:self.fid_r, thetamin:thetamax:(self.fid_theta)*1j, phimin:phimax:(self.fid_phi)*1j]
		self.grid = np.reshape(self.grid, (3,-1), order = 'C').T
		self.grid[:,0] += self.min_cell_distance #start some distance from vehicle

		nshell = self.fid_theta*(self.fid_phi) #number of grid cells per shell
		r_last = 3 #radis of line from observer to previous shell
		for i in range(1,self.fid_r):
			r_new = r_last*(1 + (np.arctan(2*np.pi/self.fid_theta)))
			self.grid[(i*nshell):((i+1)*nshell+1),0] = r_new
			r_last = r_new
		#-----------------------------------------------------------------------

		if draw == True:
			p = Points(self.s2c(self.grid), c = [0.3,0.8,0.3], r = 5)
			self.disp.append(p)


	def get_occupied_spherical(self):
		"""sweeps through scan to find occupied spherical grid cells"""

		st = time.time()

		has_pts = np.zeros(self.fid_theta*(self.fid_phi-1)*self.fid_r)

		# for rare cases with extremely large spread of points: get rid of points outisde self.rmax
		# print(self.grid[-1,0])
		# self.cloud_spherical = self.cloud_spherical[self.cloud_spherical[:,0] < self.grid[-1,0]]

		# ##(old) --------------------------------------------------------------------------------------------------
		# ###~10x slower but allows use of self.mnp parameter
		# #for each radial excursion in central surface sphere, check each cell until we find an occupied one
		# for j in range(self.fid_theta*(self.fid_phi-1)):
		# 	for i in range(self.fid_r - 2):
		# 		testpt = i*self.fid_theta*(self.fid_phi-1) + j
		# 		inside = self.find_pts_in_cell(testpt)

		# 		if np.shape(inside)[1] >= self.mnp:
		# 			has_pts[testpt] = 1
		# 			# self.draw_cell(testpt)
		# 			# self.disp.append(Points(self.cloud[inside], c = [0.4,0.8,0.3], r = 5))
		# 			break
		# # #-------------------------------------------------------------------------------------------------------

		#(new) ---------------------------------------------------------------------------------------------------
		cnt = 0
		#loop through all radial spikes
		for j in range(self.fid_theta*(self.fid_phi-1)):

			corn = self.get_corners_spherical(j)

			rmin = self.c2s(corn)[0,0]
			thetamin = self.c2s(corn)[0,1]
			thetamax = self.c2s(corn)[3,1]
			phimin = self.c2s(corn)[0,2]
			phimax = self.c2s(corn)[4,2]

			# check if spike contains any points
			in_j = np.where(np.array([self.cloud_spherical[:,0] > rmin,
									self.cloud_spherical[:,1] < thetamax,
									self.cloud_spherical[:,1] > thetamin,
									self.cloud_spherical[:,2] < phimax,
									self.cloud_spherical[:,2] > phimin
									]).all(axis = 0) == True)

			# for occupided spikes, find cell containing the first visible point
			if np.shape(in_j)[1] > 0:
				# cnt += 1
				#get point closest to the center of all the points in spike j 
				closest_radius =  np.min( self.cloud_spherical[in_j][:,0] )
				# print(closest_radius)

				#find what bin the closest point corresponds to
				# print(np.where(closest_radius > np.unique(self.grid[:,0])))
				shell_num = np.where(closest_radius > np.unique(self.grid[:,0]))[0][-1] #+ 1
				# print(shell_num)
				cell =  self.fid_theta*(self.fid_phi - 1)*shell_num + j
				# print(cell)

				# self.draw_cell(cell)
				# inside = self.find_pts_in_cell(cell)
				# self.disp.append(Points(self.cloud[inside], c = [0.4,0.8,0.3], r = 5))

				has_pts[cell] = 1

		# print("total number of useful spikes: ", cnt)
		# print("fraction of spikes containing points", cnt / (self.fid_theta*(self.fid_phi - 1))) # ONLY 20-30% !!
		#-------------------------------------------------------------------------------------------------------

		self.occupied = np.asarray(np.where(has_pts == 1))[0,:]
		print("took", time.time() - st, "s to identify occupied cells")

	def find_pts_in_cell(self,cell):
		"""get idx of points from cloud in spherical cell"""

		corn = self.get_corners_spherical(cell)

		#convert point cloud to spherical coordinates
		# self.cloud_spherical = self.c2s(self.cloud)
		# print(self.cloud_spherical)

		#get subset of points in radial band between maxr and minr (from corn)
		rmin = self.c2s(corn)[0,0]
		rmax = self.c2s(corn)[2,0]
		thetamin = self.c2s(corn)[0,1]
		thetamax = self.c2s(corn)[3,1]
		phimin = self.c2s(corn)[0,2]
		phimax = self.c2s(corn)[4,2]

		#need to fix edge case where thetamax < thetamin
		if thetamax == -np.pi:
			thetamax = np.pi

		# inside = np.where(self.cloud_spherical[:,0] < rmax )
		inside = np.where(np.array([self.cloud_spherical[:,0] < rmax,
									self.cloud_spherical[:,0] > rmin,
									self.cloud_spherical[:,1] < thetamax,
									self.cloud_spherical[:,1] > thetamin,
									self.cloud_spherical[:,2] < phimax,
									self.cloud_spherical[:,2] > phimin
									]).all(axis = 0) == True)

		# print(inside)

		return(inside)

	def get_corners_spherical(self, cell):

		n = cell + cell//(self.fid_phi - 1) #was this

		#need to account for end of ring where cells wrap around
		per_shell = self.fid_theta*(self.fid_phi - 1)
		fix =  (self.fid_phi*self.fid_theta)*((((cell)%per_shell) + (self.fid_phi-1) )//per_shell) 

		p1 = self.s2c(self.grid[n])
		p2 = self.s2c(self.grid[n + self.fid_phi - fix])
		p3 = self.s2c(self.grid[n + self.fid_theta*self.fid_phi])
		p4 = self.s2c(self.grid[n + self.fid_phi + (self.fid_theta*self.fid_phi) - fix]) 
		p5 = self.s2c(self.grid[n + 1])
		p6 = self.s2c(self.grid[n+self.fid_phi +1 - fix])
		p7 = self.s2c(self.grid[n + (self.fid_theta*self.fid_phi) + 1])
		p8 = self.s2c(self.grid[n + self.fid_phi + (self.fid_theta*self.fid_phi) +1 - fix])

		corners = np.array([p1, p2, p3, p4, p5, p6, p7, p8])

		return(corners)

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

	def c2s(self,arr):
		""" convert cartesian coordinates to spherical """

		x = arr[:,0]
		y = arr[:,1]
		z = arr[:,2]

		r = np.sqrt(x**2 + y**2 + z**2)
		phi = np.arccos(z/r)
		theta = np.arctan2(y,x)

		return(np.array([r, theta, phi]).T)

	def draw_cell(self, cell, wire = False, draw_corners = False):
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

			p1, p2, p3, p4, p5, p6, p7, p8 = self.get_corners_spherical(cell)


			if draw_corners:
				#for debug --------
				self.disp.append(Points(np.array([p1]), c = 'red', r = 10))
				self.disp.append(Points(np.array([p2]), c = 'green', r = 10))
				self.disp.append(Points(np.array([p3]), c = 'blue', r = 10))
				self.disp.append(Points(np.array([p4]), c = 'black', r = 10))
				#------------------

			# print(self.get_corners_spherical(cell))

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

			# print(arc1)
			# print(line1)

			arc3 = shapes.Arc(center = [0,0,0], point1 = p5, point2 = p6, c = 'red')			
			self.disp.append(arc3)
			arc4 = shapes.Arc(center = [0,0,0], point1 = p7, point2 = p8, c = 'red')
			self.disp.append(arc4)
			line3 = shapes.Line(p5, p7, c = 'red', lw = 1)
			self.disp.append(line3)
			line4 = shapes.Line(p6, p8, c = 'red', lw = 1)
			self.disp.append(line4)

			# self.disp.append(Points(np.array([p1]), c = "blue", r = 8))
			self.disp.append(shapes.Line(p1,p5,c = 'red', lw = 1))
			self.disp.append(shapes.Line(p2,p6,c = 'red', lw = 1))
			self.disp.append(shapes.Line(p3,p7,c = 'red', lw = 1))
			self.disp.append(shapes.Line(p4,p8,c = 'red', lw = 1))


	def draw_cloud(self):
		c = Points(self.cloud, c = [0.5, 0.5, 0.8], r = 4)
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