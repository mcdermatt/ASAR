import cv2
import numpy as np
from matplotlib import pyplot as plt
from player import Player
import time
from player import Player
from game import game

fig = plt.figure(0)
ax = fig.add_subplot()
ax.set_xlim(0,800)
ax.set_ylim(800,0)
# plt.ion()
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


#------------------------------------------------------------------

runLen = 100

data = np.zeros([runLen,103])

for i in range(runLen):

	print(i)

	g.p.place_player()
	g.p.heading = np.random.rand()*2*np.pi - np.pi
	g.p.draw()

	before = g.p.lidar

	g.fig.canvas.draw()
	plt.pause(0.01)
	g.p.remove()

	stepSize = np.random.rand()*10
	dir_rel2heading = np.random.rand()*2*np.pi - np.pi

	g.p.step(stepSize, dir_rel2heading)
	rotation = np.random.randn()*0.1
	g.p.heading = g.p.heading + rotation
	g.p.draw()

	after = g.p.lidar

	g.fig.canvas.draw()
	plt.pause(0.01)
	g.p.remove()

	g.axis.patches = []

	data[i,:50] = before
	data[i,50:100] = after
	data[i,100] = stepSize
	data[i,101] = dir_rel2heading
	data[i,102] = rotation

# print(data)
file = "data/train.npy"
np.save(file, data)