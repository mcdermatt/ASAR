import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import cv2

#TODO:
#	Add "Lidar Scan"
#	Add memory of spotted enemy positions
#	get rid of "enemy" class- should both be player (enemy makes decisions based on more basic network??)

class Player():

	def __init__(self, fig, ax, img, team = 0, FOV = 90, show_full_FOV = False):

		self.team = team
		self.score = 0
		self.alive = True
		self.weapon = 1 #single shot every 1 second
		self.FOV = np.deg2rad(FOV)
		self.enemy_pos = []
		self.axis = ax
		self.fig = fig
		self.img = img
		self.detect_chance = 1
		self.health = 150
		self.show_full_FOV = show_full_FOV

		self.scale_percent = 100 # percent of original size
		self.width = int(img.shape[1] * self.scale_percent / 100)
		self.height = int(img.shape[0] * self.scale_percent / 100)
		dim = (self.width, self.height)
		# resize image
		self.img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
		self.shot = None

		self.place_player()

	def place_player(self):

		done = False

		while done == False:
			x = int(np.random.rand()*self.width)
			y = int(np.random.rand()*self.height)

			if self.img[y,x][0] == 255 and self.img[x,y][1] == 255 and self.img[x,y][2] == 255:
				self.heading = np.random.rand()*2*np.pi - np.pi/2
				self.pos = np.array([x,y])
				# print("(x,y) = ", x, y , " heading = ", self.heading)
				done = True


	def shoot(self):

		pass

	def draw(self):

		#TODO where do I put this??		
		# if self.heading >= np.pi:
		# 	self.heading -= np.pi
		# if self.heading <= -np.pi:
		# 	self.heading += np.pi

		#draw FOV
		self.fovXendL, self.fovYendL, _ = self.RT(self.pos, self.heading + (self.FOV/2))
		self.fovXendR, self.fovYendR, _ = self.RT(self.pos, self.heading - (self.FOV/2))

		self.fovL, = self.axis.plot([self.pos[0],self.fovXendL],[self.pos[1],self.fovYendL], 'r-', lw = 1)
		self.fovR, = self.axis.plot([self.pos[0],self.fovXendR],[self.pos[1],self.fovYendR], 'r-', lw = 1)

		#test - draw inner fov
		# num = 100 
		# f = np.zeros([num,2])
		# for i in range(num):
		# 	f[i,:] = self.RT(self.pos, self.heading - (self.FOV + i/(num/2)) )
		# 	self.test, = self.axis.plot([self.pos[0],f[i,0]],[self.pos[1],f[i,1]], 'b-', lw = 1)

		#draw FOV polygon
		if self.show_full_FOV == True:
			# xy = np.array([[self.fovXendL, self.fovYendL],[self.fovXendR, self.fovYendR], [self.pos[0], self.pos[1]]])
			xy = np.array([[self.fovXendL, self.fovYendL],[self.fovXendR, self.fovYendR]])
			
			#loop through ray tracing a few times in here to get more points on the polygon
			fovfid = 100
			for i in range(fovfid):
				X, Y, _ = self.RT(self.pos,self.heading + self.FOV/2 - i*(self.FOV/fovfid))
				xy = np.concatenate((xy, np.array([[X,Y]])))

			xy = np.concatenate((xy,np.array([[self.pos[0],self.pos[1]]])))

			self.poly = Polygon(xy, closed=True, color=(0.8,0.9,1.0))
			self.axis.add_patch(self.poly)

		#draw main player body
		if self.team == 0:
			color = 'g.'
		else:
			color = 'r.'
		self.sprite, = self.axis.plot(self.pos[0],self.pos[1],color, markersize = 20)


		# write health of player
		# n = [69]
		# for i, txt in enumerate(n):
		self.h = self.axis.annotate(self.health, (self.pos[0]-20,self.pos[1]+40))

		# plt.pause(0.01)
		# self.fig.canvas.draw() #debug, this causes stuttering if called twice in a row??

	def RT(self, start, heading, endCond = np.array([0,0,0]), numPts = 100):
		"""Ray Tracing operation (through map features)

		start = starting point of ray
		heading = direction of ray 
		endCond = color of map feature to collide with 
		numPts = number of points to check on line segment"""

		stepx = start[0]
		stepy = start[1]
		stepxtrue = stepx
		stepytrue = stepy
		size = 10
		hit = False

		#TODO: repeat this until we get closer and closer to the line
		for i in range(numPts):
			#step away from starting point
			stepxtrue = stepxtrue + size*np.sin(heading)
			stepytrue = stepytrue + size*np.cos(heading)
			
			stepx = int(stepxtrue + size*np.sin(heading))
			stepy = int(stepytrue + size*np.cos(heading))
			

			if stepx >= self.height or stepx < 0:
				stepx -= int(size*np.sin(heading))
				# stepxtrue -= 10*np.sin(heading)
				break
			if stepy >= self.width or stepy < 0:
				stepy -= int(size*np.cos(heading))
				# stepytrue -= 10*np.cos(heading)
				break

			#check to make sure we didn't go outside the image
			color = self.img[stepy,stepx]
			if color[0] == endCond[0] and color[1] == endCond[1]:
				# print("hit a wall at ", stepx, stepy)
				hit = True
				break 

		# return stepx, stepy #angle between two points on grid
		return stepxtrue, stepytrue, hit #outputs smooth angles


	def remove(self):
		try:
			self.sprite.remove()
			self.fovL.remove()
			self.fovR.remove()
			self.h.remove()
		# self.test.remove()
		except:
			pass

		try:
			self.shot.remove()
		except:
			pass


class enemy():

	def __init__(self,  fig, ax, img):
		self.alive = True
		self.weapon = 1 #single shot every 1 second
		self.axis = ax
		self.fig = fig
		self.img = img

		self.scale_percent = 100 # percent of original size
		self.width = int(img.shape[1] * self.scale_percent / 100)
		self.height = int(img.shape[0] * self.scale_percent / 100)
		dim = (self.width, self.height)
		self.img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

		self.place_enemy()


	def place_enemy(self):
		done = False
		while done == False:
			x = int(np.random.rand()*self.width)
			y = int(np.random.rand()*self.height)
			if self.img[y,x][0] == 255 and self.img[x,y][1] == 255 and self.img[x,y][2] == 255:
				self.heading = np.random.rand()*360 - 180
				self.pos = np.array([x,y])
				print("(x,y) = ", x, y , " heading = ", self.heading)
				done = True

	def draw(self):

		#draw main player body
		self.sprite, = self.axis.plot(self.pos[0],self.pos[1],'r.', markersize = 20)

		plt.pause(0.01)
		self.fig.canvas.draw() 

	def remove(self):
		self.sprite.remove()
