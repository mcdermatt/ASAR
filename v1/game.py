import cv2
import numpy as np
from matplotlib import pyplot as plt
from player import Player
# from player import enemy
import time

#TODO
#	Make health system
#	Make shooting/ accuracy system
#	add noise to lidar readings
#		include player position data in lidar? -might be expensive
#	be able to add an arbitrary number of enemies

#BUGS
#	Line of sight (yellow) persists if player sees two enemies at once

class game():

	def __init__(self,fig,ax,img,numEnemies = 1):

		self.axis = ax
		self.fig = fig

		self.players = []
		self.p = Player(fig,ax,img, team = 0, FOV=60, show_full_FOV = True)
		self.players.append(self.p)

		self.enemies = []

		self.numEnemies = numEnemies
		for i in range(self.numEnemies):
			setattr(self, "e" + str(i), Player(fig,ax,img,team=1,FOV=90))

			self.enemies.append( getattr(self, "e" + str(i)) )
		
		# self.e1 = Player(fig,ax,img, team = 1, FOV=90)
		# self.enemies.append(self.e1)
		# self.e2 = Player(fig,ax,img, team = 1, FOV=90)
		# self.enemies.append(self.e2)

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
		# self.e1.remove()

	def look_for_enemy(self):

		for m in range(len(self.players)):
			p = self.players[m]
			
			# for e in self.enemies: #creates ugly strings like <player.Player object at 0x0000...>
			for n in range(len(self.enemies)):
				e = self.enemies[n] 

				#get rid of existing shooting lines
				# try:
				# 	getattr(self.p, str(e) + "shot").remove()
				# except:
				# 	# print("failed")
				# 	pass
				
				#check if player is looking at enemy
				bearing = np.arctan2((p.pos[0]-e.pos[0]),(p.pos[1]-e.pos[1]))

				if bearing >= np.pi:
					bearing -= np.pi
				if bearing <= -np.pi:
					bearing += np.pi

				#draw bearing line from player pointing towards each enemy
				# self.axis.plot([p.pos[0],p.pos[0]-np.sin(bearing)*100],[p.pos[1],p.pos[1]-np.cos(bearing)*100],'k-')

				#if bearing is within FOV
				if (abs(np.sin(self.p.heading + np.pi) - np.sin(bearing)) < np.sin(p.FOV/2)) and (abs(np.cos(self.p.heading + np.pi) - np.cos(bearing)) < np.sin(p.FOV/2)) :
					# print("bearing = ", bearing)
					# print("heading = ", self.p.heading)

					#check if enemy is in line of sight
					#get xy cords of nearest wall in line of sight
					x, y, _ = p.RT(p.pos, bearing + np.pi) #bearing is off by pi... debug??

					#for debug- show where line of sight collides with obstacles
					# t = self.axis.plot(x,y,'r.')

					#get distance to enemy and wall
					d_to_enemy = np.sqrt((p.pos[0]-e.pos[0])**2 + (p.pos[1]-e.pos[1])**2) #good
					d_to_wall = np.sqrt(((p.pos[1]-y)**2) + (p.pos[0]-x)**2) #not right??

					# take shot if distance to enemy is closer than the nearest obstacle in that direction
					if d_to_enemy < d_to_wall:

						#shooting line-----------------
						#TODO: need to remove if more than one enemy...
						# setattr(self.p, str(e) + "shot", self.axis.plot([p.pos[0],e.pos[0]],[p.pos[1],e.pos[1]],'y-', lw = 2))
						
						#was this
						# self.p.shot, = self.axis.plot([p.pos[0],e.pos[0]],[p.pos[1],e.pos[1]],'y-', lw = 2) 

						if e.alive == True:
							p.score += 10 
							print("score: ", p.score)
							e.health -= 10

	def run(self):
		"""debug function for now"""

		# self.e1.pos = np.array([400,500])
		self.p.pos = np.array([500,700])

		self.p.heading = np.pi

		for i in range(400):

			# self.p.heading = np.cos(i/10) * 2

			#move in circle
			# self.p.pos[0] = 400 + 300*np.cos(i/5)
			# self.p.pos[1] = 400 + 300*np.sin(i/15)

			self.p.heading += np.random.randn()*0.25
			self.p.step(size=10)
			# self.e1.heading += np.random.randn()*0.25
			# self.e1.step(size=5)

			self.look_for_enemy()
			for e in self.enemies:
				e.heading += np.random.randn()*0.25
				e.step(size=5)
				
				if e.health <= 0:
					e.alive = False

				if e.alive == True:
					e.draw()

			self.p.draw()
			self.fig.canvas.draw()

			time.sleep(0.01)
			self.p.remove()
			for e in self.enemies:
				e.remove()

			# self.e2.remove()	
			self.axis.patches = []
