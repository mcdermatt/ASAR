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

		self.drawVoxel(4)

		self.findPtsInVox(4)

		c = self.getCentroid(4)
		self.ax.plot(c[0],c[1],'bx')


		p = PatchCollection(self.patches, alpha=0.4)
		self.ax.add_collection(p)
		plt.show()


	def initNodes(self):
		""" set initial positions of nodes for voxel grid """

		node_boundaries = np.array([-10, 10, -10, 10]) #[minx, maxx, miny, maxy]
		node_fidelity = 4 #this many nodes in x and y 

		self.Nodes = np.zeros([node_fidelity**2,2])

		self.Nodes[:,0] = np.tile(np.linspace(node_boundaries[0], node_boundaries[1], node_fidelity), node_fidelity)

		ypos = np.linspace(node_boundaries[3], node_boundaries[2], node_fidelity)
		for i in range(1,node_fidelity+1):
			self.Nodes[(i-1)*node_fidelity:i*node_fidelity,1] = ypos[i-1]

		print("created grid with", (node_fidelity-1)**2, "voxels")


	def drawNodes(self):
		self.ax.plot(self.Nodes[:,0], self.Nodes[:,1], 'b.')

	def getCorners(self,voxid):
		""" gets corner nodes bounding voxel # voxid """

		node_fidelity = 4  

		tl = voxid + voxid//(node_fidelity - 1) #top left
		tr = tl+1
		br = tr + node_fidelity
		bl = tl + node_fidelity

		corners = np.array([self.Nodes[tl,:],
							self.Nodes[tr,:], 
							self.Nodes[br,:], 
							self.Nodes[bl,:]])
		return corners

	def drawVoxel(self, voxid):
		""" draws polygon using corners"""
		corners = self.getCorners(voxid)

		poly = Polygon(corners, True)
		self.patches.append(poly)

	def genDemoCloud(self):
		"""generate simple structured point cloud for testing """

		self.cloud = np.random.randn(10,2)
		self.cloud[:,0] += np.linspace(-3,3,10)
		self.cloud[:,1] += np.linspace(1,4,10)


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

		# print(c)

		return c