import cv2
import numpy as np
from matplotlib import pyplot as plt
from player import Player
from player import enemy
import time

class game():

	def __init__(self,fig,ax,img):

		self.players = []
		self.p = Player(fig,ax,img, team = 0, FOV=90)
		self.players.append(self.p)

		self.enemies = []
		self.e1 = Player(fig,ax,img, team = 1, FOV=90)
		self.enemies.append(self.e1)


		#debug
		self.p.pos = np.array([600,600])
		self.p.heading = -np.pi/2 + 0.1
		self.e1.pos = np.array([500,600])


		self.draw()


	def draw(self):

		try:
			self.p.remove()
			self.e1.remove()
		except:
			pass

		self.p.draw()
		self.e1.draw()

		self.look_for_enemy()

		time.sleep(1)
		self.p.remove()
		self.e1.remove()

	def look_for_enemy(self):

		for p in self.players:
			for e in self.enemies:
				
				#check if player is looking at enemy
				bearing = abs(np.arctan2((p.pos[0]-e.pos[0]),(-p.pos[1]+e.pos[1]))) - abs(p.heading)
				print(bearing)
				if bearing < p.FOV/2:
					#check if enemy is in line of sight
					randang = np.random.randn()*np.pi

					#DEBUG - can't hit enemy because red dot not actually on img
					x, y, hit = p.RT(p.pos, randang, endCond = np.array([255,0,0]))
					print(hit)

					if hit:
						self.axis.plot([self.pos[0],x],[self.pos[1],y],'g-')
						print("angle that hit: ",randang)