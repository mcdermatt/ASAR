import cv2
import numpy as np
from matplotlib import pyplot as plt
from player import Player
import time

from player import Player
from game import game

#input data from enviornment:
#	lidar points (x100)
#		in later version, include enemy body in lidar (currently just has map)
#	heading
#	global position
#	seen enemy position(s)
#	how many seconds since enemy was last seen??
#	player health
#	NEED TO REPRESENT VELOCITY
#		save past n frames of Lidar??


# Netork Architecture- DDPG vs RNN
#	DDPG
#		more complex network structures
#		run from scratch each time
#
#	RNN
#		single network (LSTM/GRU in TF)
#		REQUIRES CORPUS (bad)


# Reward shaping
#	what do we want here?

#init ------------------------------------------------------------------

	#init map

	#place player and enemy

	#init network W&B 
	#	start from checkpoint or no

	# get initial state measurements from enviornment

fig = plt.figure(0)
ax = fig.add_subplot()
ax.set_xlim(0,800)
ax.set_ylim(800,0)

plt.ion()
plt.axis('off')
fig.show()
fig.patch.set_facecolor('xkcd:greyish blue')
fig.canvas.draw()

img = cv2.imread('assets/map1.png')

scale_percent = 100 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
ax.imshow(img, cmap = 'gray', interpolation = 'bicubic')

g = game(fig,ax,img,numEnemies=0)


#main ------------------------------------------------------------------

g.p.fovfid = 100
g.p.pos = np.array([500,700])

g.p.heading = np.pi

for i in range(400):

	g.p.heading += np.random.randn()*0.25
	g.p.step(size=10)
	g.look_for_enemy()
	for e in g.enemies:
		e.heading += np.random.randn()*0.25
		e.step(size=5)
		
		if e.health <= 0:
			e.alive = False

		if e.alive == True:
			e.draw()

	g.p.draw()
	g.fig.canvas.draw()

	plt.pause(0.01)

	g.p.remove()
	for e in g.enemies:
		e.remove()
	g.axis.patches = []