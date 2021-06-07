import numpy as np
from matplotlib import pyplot as plt
import cv2

#TODO 

# FOV limits
#	fix scale map image -> for performance


class Player():

	def __init__(self, fig, ax, img, FOV = 90):

		self.alive = True
		self.weapon = 0 #single shot every 1 second

		self.FOV = np.deg2rad(FOV)
		self.enemy_pos = None

		self.heading = 0
		self.pos = np.array([600,700])

		self.axis = ax
		self.fig = fig
		self.img = img

		self.scale_percent = 100 # percent of original size
		self.width = int(img.shape[1] * self.scale_percent / 100)
		self.height = int(img.shape[0] * self.scale_percent / 100)
		dim = (self.width, self.height)
		# resize image
		self.img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)



	def shoot(self):

		pass

	def draw(self):
		
		#draw FOV
		fovXendL, fovYendL = self.RT(self.pos, self.heading + (self.FOV/2))
		fovXendR, fovYendR = self.RT(self.pos, self.heading - (self.FOV/2))
		self.fovL, = self.axis.plot([self.pos[0],fovXendL],[self.pos[1],fovYendL], 'r-', lw = 2)
		self.fovR, = self.axis.plot([self.pos[0],fovXendR],[self.pos[1],fovYendR], 'r-', lw = 2)

		#test - draw inner fov
		# num = 10 
		# f = np.zeros([10,2])
		# for i in range(num):
		# 	f[i,:] = self.RT(self.pos, self.heading - (self.FOV + i/(num/2)) )
		# 	test, = self.axis.plot([self.pos[0],f[i,0]],[self.pos[1],f[i,1]], 'b-', lw = 1)

		#draw main player body
		self.sprite, = self.axis.plot(self.pos[0],self.pos[1],'g.', markersize = 20)

		plt.pause(0.01)
		self.fig.canvas.draw()

	def RT(self, start, heading, endCond = np.array([0,0,0]), numPts = 100):
		"""Ray Tracing operation"""

		stepx = start[0]
		stepy = start[1]

		for i in range(numPts):
			#step away from starting point
			stepx = stepx + int(10*np.sin(heading))
			stepy = stepy + int(10*np.cos(heading))
			
			if stepx >= self.height or stepx < 0:
				stepx -= int(10*np.sin(heading))
				break
			if stepy >= self.width or stepy < 0:
				stepy -= int(10*np.cos(heading))
				break

			#check to make sure we didn't go outside the image
			color = self.img[stepy,stepx]
			if color[0] == endCond[0] and color[1] == endCond[1]:
				# print("hit a wall at ", stepx, stepy)
				break 

		return stepx, stepy

	def remove(self):
		self.sprite.remove()
		self.fovL.remove()
		self.fovR.remove()