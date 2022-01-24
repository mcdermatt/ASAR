import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.path as mplPath

class divide():

	"""numpy prototype of gradient descent voxel subdivision in 2D"""

	def __init__(self, fig, ax, cloud = None):

		self.minNumPts = 3

		self.fig = fig
		self.ax = ax
		self.patches = []

		if cloud == None:
			self.genDemoCloud()
		else:
			self.cloud = cloud

		self.initNodes()

		# self.Nodes[5] += np.array([3,-2]) #test-> drag around a node to verify centroid updates

		self.drawNodes()


		# self.findPtsInVox(4)

		c = self.getCentroid(4)
		self.ax.plot(c[0],c[1],'bx')

		self.main()

		plt.show()


	def main(self):
		""" Main loop: iteratively adjusts placement of nodes to optimize a score function """

		numIter = 1
		numVox = (self.node_fidelity-1)**2

		for i in range(numIter):
			for j in range(numVox):

				pts = self.findPtsInVox(j)

				self.drawVoxel(j, npts = np.shape(pts)[0])

				#only consider voxels with sufficient number of points

				if np.shape(pts)[0] > self.minNumPts:
					mu, sigma = self.fit_gaussian(pts)
					# print(sigma)



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
		self.node_fidelity = 4 #this many nodes in x and y 

		self.Nodes = np.zeros([self.node_fidelity**2,2])

		self.Nodes[:,0] = np.tile(np.linspace(node_boundaries[0], node_boundaries[1], self.node_fidelity), self.node_fidelity)

		ypos = np.linspace(node_boundaries[3], node_boundaries[2], self.node_fidelity)
		for i in range(1,self.node_fidelity+1):
			self.Nodes[(i-1)*self.node_fidelity:i*self.node_fidelity,1] = ypos[i-1]

		print("created grid with", (self.node_fidelity-1)**2, "voxels")


	def drawNodes(self):
		self.ax.plot(self.Nodes[:,0], self.Nodes[:,1], 'b.')

	def getCorners(self,voxid):
		""" gets corner nodes bounding voxel # voxid """

		self.node_fidelity = 4  

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
			p.set_color([1 - npts/(np.shape(self.cloud)[0]) ,1 - npts/(np.shape(self.cloud)[0]) ,1 ])

		self.ax.add_collection(p)

	def genDemoCloud(self):
		"""generate simple structured point cloud for testing """

		npts = 50
		self.cloud = 0.5*np.random.randn(npts,2)
		self.cloud[:,0] += np.linspace(-8,8,npts)
		self.cloud[:,1] += np.linspace(1,4,npts)


		self.ax.plot(self.cloud[:,0], self.cloud[:,1], 'r.')

	def findPtsInVox(self, voxid):
		""" find point IDs in scan that lie inside a given voxid """

		corners = self.getCorners(voxid)
		vox = mplPath.Path(corners)
		isInside = vox.contains_points(self.cloud)
		inside = self.cloud[isInside]
		self.ax.plot(inside[:,0],inside[:,1], 'ro')

		return inside

	def getCentroid(self, voxid):
		""" returns centroid of polygon specified by voxid """

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