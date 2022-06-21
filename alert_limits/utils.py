from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Ellipse, Arc
from matplotlib.collections import PatchCollection

class scene():
	"""Define class contianing all necessary parameters to calculate and visualize
	   the alert limit formulation defined in Olivier's paper"""

	def __init__(self, pos = np.array([-3,2]), theta = -60, road_width = 5, road_radius = 10):

		self.pos = pos #location of center of vehicle
		self.theta = theta #rotation of vehicle relative to global North/ East
		self.road_width = road_width

		#init vehicle params
		self.v_w = 2 #vehicle width
		self.v_h = 4 #vehicle height

		#init structure for storing "patches". These are used as 
		self.patches = []

		self.init_scene()

		#not sure if it going to make sense to automatically draw on scene population
		#   going to leave this here for now, remove if it makes sense to do so
		# self.draw()

	def init_scene(self):

		#init figure and axis
		self.fig, self.ax = plt.subplots()

		#set title and axis labels
		self.ax.set_title("Alert Limits")
		self.ax.set_xlabel("East")
		self.ax.set_ylabel("North")
		self.ax.set_aspect("equal")
		self.ax.set_xlim([-10,10])
		self.ax.set_ylim([-10,10])

		#define and draw road ----------
		outer_w = 30
		outer_h = 30
		outer_pos = (5,-5)
		road_outer = Arc(outer_pos, outer_w, outer_h, theta1 = 90, theta2 = 180)
		self.ax.add_patch(road_outer)

		#TODO: define inner bound using similar variables to outer bound
		#	   define this in a way that maintains constant road width
		road_inner = Arc((10,-10), 20, 20, theta1 = 90, theta2 = 180)
		self.ax.add_patch(road_inner)
		#-------------------------------

		self.add_ebb()
		self.add_alert_limit()
		self.add_car()

	def add_car(self):
		""" Adds main vehicle to the scene """

		#rectangles are defined by a corner (rather than the center) so we're going to be lazy here 
		#  and just draw 4 corner-defined rectangles

		#front driver side
		fd = Rectangle(self.pos, -self.v_w/2, self.v_h/2, angle = self.theta, facecolor = (0.3,0.3,0.5), edgecolor = None)
		self.ax.add_patch(fd)
		#front passanger side
		fp = Rectangle(self.pos, self.v_w/2, self.v_h/2, angle = self.theta, facecolor = (0.3,0.3,0.5), edgecolor = None)
		self.ax.add_patch(fp)
		#rear driver side
		rd = Rectangle(self.pos, -self.v_w/2, -self.v_h/2, angle = self.theta, facecolor = (0.3,0.3,0.5), edgecolor = None)
		self.ax.add_patch(rd)
		#rear passanger side
		rp = Rectangle(self.pos, self.v_w/2, -self.v_h/2, angle = self.theta, facecolor = (0.3,0.3,0.5), edgecolor = None)
		self.ax.add_patch(rp)

		self.ax.plot(self.pos[0], self.pos[1], 'r.')


	def add_alert_limit(self):
		""" adds alert limits to vehicle """

		R = rot(self.theta) #get rotation matrix

		#front driver side
		corner_pos = self.pos + R.dot(np.array([-self.v_w/2, self.v_h/2]))
		fd = Ellipse(corner_pos, width = 1, height = 1) 

		self.ax.add_patch(fd)
		
	def add_ebb(self):
		""" adds extended bonding box """

		#TODO: implement this here -----------------

		#-------------------------------------------

		pass

	def draw(self):
		""" draws scene using all patches stored iside self.patches """
		pc = PatchCollection(self.patches)
		self.ax.add_collection(pc)

	def check_alert_limit(self, corner):
		""" Check if altert limit for specified corner is located within 
		the bounds of the road  """

		#TODO: add in meat of this function here
		#	easiest way to do this is to use the pre-defined equations of the circular
		#`	road bounds to test if suspected point on alert limit is within outer road edge
		#   circle but outside lower road edge circle

		pass

def rot(ang):
	""" Generates rotation matrix from angle <ang> """

	ang = np.deg2rad(ang)

	R = np.array([[np.cos(ang), -np.sin(ang)],
				  [np.sin(ang), np.cos(ang)]])
	return R