import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath

class divide():

	"""numpy prototype of gradient descent voxel subdivision in 2D"""

	def __init__(self, fig, ax, cloud = None, n = 10):

		np.random.seed(187)

		self.minNumPts = 3
		self.node_fidelity = 12 #creates an NxN grid of nodes
		self.numIter = n #number of iterations to loop through


		self.fig = fig
		self.ax = ax
		self.patches = []

		if cloud == None:
			self.genDemoCloud()
		else:
			self.cloud = cloud

		self.initNodes()
		# self.drawNodes()

		self.main()

		# plt.show()


	def main(self):
		""" Main loop: iteratively adjusts placement of nodes to optimize a score function """

		numVox = (self.node_fidelity-1)**2

		for i in range(self.numIter):

			#initialize score vector
			self.corrections = np.zeros(np.shape(self.Nodes))

			for j in range(numVox):

				pts = self.findPtsInVox(j)
				#only draw on last
				if i == self.numIter - 1:
					self.drawVoxel(j, npts = np.shape(pts)[0]) #specifing number of points effects shading of voxel

				#only consider voxels with sufficient number of points
				if np.shape(pts)[0] > self.minNumPts:
					#get mean (mu) and covariance (sigma) of distribution of points within distribution
					mu, sigma = self.fit_gaussian(pts)
					# self.ax.plot(mu[0], mu[1], 'rx') #plot centers of per-voxel distributions

					eig = np.linalg.eig(sigma)
					eigenval = eig[0]
					eigenvec = eig[1]

					rot = -np.rad2deg(np.arctan(eigenvec[0,1]/eigenvec[0,0]))
					width = 4*np.sqrt(abs(eigenval[0]))
					height = 4*np.sqrt(abs(eigenval[1]))
					ell = Ellipse((mu[0],mu[1]),width, height, angle = rot, fill = True, color = [1, 0.5,0.5, 0.7])
					self.ax.add_patch(ell)

					#calculate centroid of the voxel
					c = self.getCentroid(voxid = j)
					# self.ax.plot(c[0],c[1],'bx') #plot centroids

					# calcualte distance between voxel centroid and distribution center
					d0j = np.sqrt( (c[0] - mu[0])**2 + (c[1] - mu[1])**2) 
					# print(d0j)

					#Strategy 1 ----------------------------------------------------------
					#for each corner of the voxel, drag the node by a small perterbation
					#  in the x and y directions, and re-calculate the distane betwen 
					#  voxel centroid and distribution centers
					
					#get current locations of corners
					corners = self.getCorners(j)
					# print(corners)

					#loop through the 4 corners
					for k in range(4):

						eps = 0.01 #arbitrarily small displacement

						#move slightly in the x-direction and calculate change in score
						corners_temp = corners.copy()
						corners_temp[k,0] += eps
						c_dx_jk = self.getCentroid(nodes = corners_temp)#centroid of voxel j after dx is applied to node k
						ddkx = d0j - np.sqrt( (c_dx_jk[0] - mu[0])**2 + (c_dx_jk[1] - mu[1])**2)
						# print(ddkx)

						#move slightly in the y-direction and note change in score
						corners_temp = corners.copy()
						corners_temp[k,1] += eps
						c_dy_jk = self.getCentroid(nodes = corners_temp)#centroid of voxel j after dy is applied to node k
						ddky = d0j - np.sqrt( (c_dy_jk[0] - mu[0])**2 + (c_dy_jk[1] - mu[1])**2)

						#identify which element in self.Nodes corresponds to the corner coordinaces
						n = np.where((self.Nodes == corners[k]).all(axis = 1) == True)
						# print(n)

						#use these deltas to contribute to an overall score vector for each node												
						#weight these contributons as a func. of the number of points in the cell?
						self.corrections[n,0] += ddkx*np.shape(pts)[0]
						self.corrections[n,1] += ddky*np.shape(pts)[0]


					#-----------------------------------------------------------------

			#apply score vector to nodes to adjust their positions
			self.Nodes += self.corrections
			# self.drawNodes()

			#TODO: create a test to make sure voxels don't fold up on top of one another
		

			#TODO: figure out how to dynamically adjust the extended axis pruning threshold
			#		could try to inscribe a circle inside each polygon??


	def fit_gaussian(self, pts, draw = True):
		""" calaculates covaraince of group of points """

		mu_x = np.mean(pts[:,0])
		mu_y = np.mean(pts[:,1])
		mu = np.array([mu_x, mu_y])

		std_x = np.sum( (pts[:,0] - mu_x)*(pts[:,0] - mu_x) ) / (np.shape(pts)[0] - 1 )
		std_y = np.sum( (pts[:,1] - mu_y)*(pts[:,1] - mu_y) ) / (np.shape(pts)[0] - 1 )
		std_xy= np.sum((pts[:,0] - mu_x)*(pts[:,1] - mu_y) ) / (np.shape(pts)[0] - 1)
		sigma = np.array([[std_x,  std_xy],
						  [std_xy, std_y]])


		if draw == True:
			#draw ellipse			
			pass

		return(mu, sigma)

	def initNodes(self):
		""" set initial positions of nodes for voxel grid """

		node_boundaries = np.array([-10, 10, -10, 10]) #[minx, maxx, miny, maxy]

		self.Nodes = np.zeros([self.node_fidelity**2,2])

		self.Nodes[:,0] = np.tile(np.linspace(node_boundaries[0], node_boundaries[1], self.node_fidelity), self.node_fidelity)

		ypos = np.linspace(node_boundaries[3], node_boundaries[2], self.node_fidelity)
		for i in range(1,self.node_fidelity+1):
			self.Nodes[(i-1)*self.node_fidelity:i*self.node_fidelity,1] = ypos[i-1]

		# print("created grid with", (self.node_fidelity-1)**2, "voxels")


	def drawNodes(self):
		self.ax.plot(self.Nodes[:,0], self.Nodes[:,1], 'b.')

	def getCorners(self,voxid):
		""" gets corner nodes bounding voxel # voxid """

		tl = voxid + voxid//(self.node_fidelity - 1) #top left
		tr = tl+1
		br = tr + self.node_fidelity
		bl = tl + self.node_fidelity

		corners = np.array([self.Nodes[tl,:],
							self.Nodes[tr,:], 
							self.Nodes[br,:], 
							self.Nodes[bl,:]])
		return corners

	def drawVoxel(self, voxid, npts = -1):
		""" draws polygon using corners"""
		corners = self.getCorners(voxid)

		poly = Polygon(corners, closed = True)

		self.patches = []
		self.patches.append(poly)
		p = PatchCollection(self.patches, alpha=0.4)

		if npts != -1:
			p.set_color([(0.6 - npts/(np.shape(self.cloud)[0]))**2 , (0.6 - npts/(np.shape(self.cloud)[0]))**2 ,1 ])

		self.ax.add_collection(p)

	def genDemoCloud(self):
		"""generate simple structured point cloud for testing """

		npts = 1000 #100
		self.cloud = 0.25*np.random.randn(npts,2)
		self.cloud[:npts//4,0] += np.linspace(-8,8,npts//4)
		self.cloud[:npts//4,1] += np.linspace(1,4,npts//4)

		self.cloud[npts//4:npts//2,0] += np.linspace(1,-4, npts//4)
		self.cloud[npts//4:npts//2,1] += np.linspace(-8, 4, npts//4)

		self.cloud[npts//2:,0] += np.linspace(-5,5, npts//2)
		self.cloud[npts//2:,1] += np.linspace(-3, -2, npts//2)


		self.ax.plot(self.cloud[:,0], self.cloud[:,1], 'r.')

	def findPtsInVox(self, voxid):
		""" find point IDs in scan that lie inside a given voxid """

		corners = self.getCorners(voxid)
		vox = mplPath.Path(corners)
		isInside = vox.contains_points(self.cloud)
		inside = self.cloud[isInside]

		#highlight points in voxel
		# self.ax.plot(inside[:,0],inside[:,1], 'ro')

		return inside

	def getCentroid(self, voxid = None, nodes = None):
		""" returns centroid of polygon specified by voxid """

		if voxid == None:
			corners = nodes
		else:
			corners = self.getCorners(voxid)
		tl = corners[0,:]
		tr = corners[1,:]
		br = corners[2,:]
		bl = corners[3,:]

		#centroid of a tiangle is the average of the three vertices
		center1 = (tl + tr + bl)/3
		center2 = (bl + br + tr)/3

		#using heron's formula A = sqrt(s(s-a)(s-b)(s-c))
		a1 = np.sqrt( (tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
		b1 = np.sqrt( (tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
		c1 = np.sqrt( (bl[0] - tr[0])**2 + (bl[1] - tr[1])**2 )
		s1 = (a1 + b1 + c1) / 2
		area1 = np.sqrt(s1*(s1-a1)*(s1-b1)*(s1-c1))
		
		a2 = np.sqrt( (bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
		b2 = np.sqrt( (bl[0] - tr[0])**2 + (bl[1] - tr[1])**2 )
		c2 = np.sqrt( (br[0] - tr[0])**2 + (br[1] - tr[1])**2 )
		s2 = (a2 + b2 + c2) / 2
		area2 = np.sqrt(s2*(s2-a2)*(s2-b2)*(s2-c2))

		#centroid of the combined quadralateral is the weighted sum of the two constituant triangles
		c = center1*(area1/(area1+area2)) + center2*(area2/(area1+ area2))

		return c