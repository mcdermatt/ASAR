import cv2
import numpy as np
from matplotlib import pyplot as plt
from player import Player
from player import enemy
import time

class game():

	def __init__(self,fig,ax,img):

		self.axis = ax
		self.fig = fig

		self.players = []
		self.p = Player(fig,ax,img, team = 0, FOV=60)
		self.players.append(self.p)

		self.enemies = []
		self.e1 = Player(fig,ax,img, team = 1, FOV=90)
		self.enemies.append(self.e1)


		# #debug
		# self.p.pos = np.array([600,600])
		# self.p.heading = -np.pi/2 + .5
		# self.e1.pos = np.array([500,600])


		self.draw()


	def draw(self):

		try:
			self.p.remove()
			self.e1.remove()
		except:
			pass

		# self.p.draw()
		# self.e1.draw()

		self.look_for_enemy()

		time.sleep(1)
		self.p.remove()
		self.e1.remove()

	def look_for_enemy(self):

		for p in self.players:
			for e in self.enemies:
				
				#check if player is looking at enemy
				bearing = np.arctan2((p.pos[0]-e.pos[0]),(p.pos[1]-e.pos[1]))

				if bearing >= np.pi:
					bearing -= np.pi
				if bearing <= -np.pi:
					bearing += np.pi

				#draw bearing line from player pointing towards enemy
				# self.axis.plot([p.pos[0],p.pos[0]-np.sin(bearing)*100],[p.pos[1],p.pos[1]-np.cos(bearing)*100],'k-')

				# if abs(self.p.heading - bearing) < p.FOV/2: # was this
				if (abs(np.sin(self.p.heading + np.pi) - np.sin(bearing)) < np.sin(p.FOV/2)) and (abs(np.cos(self.p.heading + np.pi) - np.cos(bearing)) < np.sin(p.FOV/2)) :
					# print("bearing = ", bearing)
					# print("heading = ", self.p.heading)

					#check if enemy is in line of sight
					#get xy cords of nearest wall in line of sight
					x, y, _ = p.RT(p.pos, bearing, endCond = np.array([255,0,0]))

					#get distance to enemy
					d_to_enemy = np.sqrt((p.pos[0]-e.pos[0])**2 + (p.pos[1]-e.pos[1])**2)

					# draw shooting line if distance to enemy is closer than the nearest obstacle in that direction
					if d_to_enemy < np.sqrt(((p.pos[1]-x)**2) + (p.pos[0]-y)**2):
						self.axis.plot([p.pos[0],e.pos[0]],[p.pos[1],e.pos[1]],'y-', lw = 2)


	def run(self):
		"""debug function for now"""

		# self.e1.pos = np.array([400,500])
		self.e1.draw()

		for i in range(400):

			# self.p.heading = np.cos(i/200)
			self.p.pos[0] = 400 + 300*np.cos(i/5)
			self.p.pos[1] = 400 + 300*np.sin(i/5)
	
			self.p.draw()
			self.look_for_enemy()
			time.sleep(0.01)
			self.p.remove()	