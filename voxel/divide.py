import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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

		self.drawNodes()

		self.drawVoxel(9)

		p = PatchCollection(self.patches, alpha=0.4)
		self.ax.add_collection(p)
		plt.show()


	def initNodes(self):
		""" set initial positions of nodes for voxel grid """
		# self.Nodes = np.linspace(1,10,10)

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


	def drawVoxel(self, voxid):
		""" draws polygon around enumerated voxel """

		node_fidelity = 4  

		corners = np.array([self.Nodes[voxid,:],
							self.Nodes[voxid+ 1,:], 
							self.Nodes[voxid+ node_fidelity + 1,:], 
							self.Nodes[voxid+ node_fidelity,:]])

		poly = Polygon(corners, True)

		self.patches.append(poly)

	def genDemoCloud(self):
		"""generate simple structured point cloud for testing """

		self.cloud = np.random.randn(10,2)
		self.cloud[:,0] += np.linspace(-3,3,10)
		self.cloud[:,1] += np.linspace(1,4,10)


		self.ax.plot(self.cloud[:,0], self.cloud[:,1], 'r.')
