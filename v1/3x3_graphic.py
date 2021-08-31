import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *
#for generating scans
from player import Player 
from game import game


#TODO
#	get noisy scans
#	only consider points that are hitters


fig, ax = plt.subplots(nrows = 3, ncols = 3)
# fig.tight_layout() #meh
fig.subplots_adjust(left = 0.01, right = 0.99, hspace = 0.001, wspace = 0.001) #slightly better

#set labels on top
ax[0,0].set_title("Road Sign")
ax[0,1].set_title("Car")
ax[0,2].set_title("Extended Wall")
#show labels on side
ax[0,0].set_ylabel("Actual Object (above)")
ax[1,0].set_ylabel("Noise Free Scan")
ax[2,0].set_ylabel("Noisy Scan")

for i in range(3):
	for j in range(3):
		ax[i,j].axes.xaxis.set_ticks([])
		ax[i,j].axes.yaxis.set_ticks([])
		if j !=2:
			ax[i,j].set_aspect('equal')
		# else:
		# 	ax[i,j].set_aspect('0.333')


# show actual objects -----------------------------------------
sign_file = "assets/sign.png"
sign_img = cv2.imread(sign_file)
ax[0,0].imshow(sign_img,cmap="gray", interpolation = 'bicubic')

car_file = "assets/car.png"
car_img = cv2.imread(car_file)
ax[0,1].imshow(car_img,cmap="gray", interpolation = 'bicubic')

wall_file = "assets/wall.png"
wall_img = cv2.imread(wall_file)
ax[0,2].imshow(wall_img,cmap="gray", interpolation = 'bicubic')
# -------------------------------------------------------------


# Generate first scans ----------------------------------------
#sign
g0 = game(fig, ax[0,0], sign_img)#cv2.flip(sign_img,0))
g0.p.fovfid = 100
g0.p.pos = np.array([400,800])
g0.p.heading = np.pi #0 #looking straight up
g0hit = g0.p.draw(ignore_boundary = True)



#low noise scan
pp0 = draw_scan(g0.p.lidar, fig, ax[1,0], FOV = 60, ignore_boundary = True, hitters = g0hit)
ax[1,0].axes.set_aspect('equal')
ax[1,0].set_xlim(-400,400)
ax[1,0].set_ylim(0,800)

#car
g1 = game(fig, ax[0,1], car_img)#cv2.flip(car_img,0))
g1.p.fovfid = 100
g1.p.pos = np.array([400,800])
g1.p.heading = np.pi #0 #looking straight up
g1hit = g1.p.draw(ignore_boundary = True)
#low noise scan
pp1 = draw_scan(g1.p.lidar, fig, ax[1,1], FOV = 60, ignore_boundary = True, hitters = g1hit)
ax[1,1].axes.set_aspect('equal')
ax[1,1].set_xlim(-400,400)
ax[1,1].set_ylim(0,800)
#TODO: get elements of pp1 where g1hit==1

#wall
g2 = game(fig, ax[0,2], wall_img)#cv2.flip(wall_img,0)
g2.p.fovfid = 100
g2.p.pos = np.array([1200,800])
g2.p.heading = np.pi #0 #looking straight up
g2hit = g2.p.draw(ignore_boundary = True)
#low noise scan
pp2 = draw_scan(g2.p.lidar, fig, ax[1,2], FOV = 60, ignore_boundary = True, hitters = g2hit)
ax[1,2].axes.set_aspect('equal')
ax[1,2].set_xlim(-1200,1200)
ax[1,2].set_ylim(0,800)
#-------------------------------------------------------------

# Generate seccond scans ----------------------------------------
#sign
g0.p.noiseScale = 10
g0hit2 = g0.p.draw(show = False, ignore_boundary = True)
#low noise scan
pp0 = draw_scan(g0.p.lidar, fig, ax[2,0], FOV = 60, ignore_boundary = True, hitters = g0hit2)
ax[2,0].axes.set_aspect('equal')
ax[2,0].set_xlim(-400,400)
ax[2,0].set_ylim(0,800)
print(g0hit2)


# #car
# g1 = game(fig, ax[0,1], car_img)#cv2.flip(car_img,0))
# g1.p.fovfid = 100
# g1.p.pos = np.array([400,800])
# g1.p.heading = np.pi #0 #looking straight up
# g1hit = g1.p.draw(ignore_boundary = True)
# #low noise scan
# pp1 = draw_scan(g1.p.lidar, fig, ax[1,1], FOV = 60, ignore_boundary = True, hitters = g1hit)
# ax[1,1].axes.set_aspect('equal')
# ax[1,1].set_xlim(-400,400)
# ax[1,1].set_ylim(0,800)
# #TODO: get elements of pp1 where g1hit==1

# #wall
# g2 = game(fig, ax[0,2], wall_img)#cv2.flip(wall_img,0)
# g2.p.fovfid = 100
# g2.p.pos = np.array([1200,800])
# g2.p.heading = np.pi #0 #looking straight up
# g2hit = g2.p.draw(ignore_boundary = True)
# #low noise scan
# pp2 = draw_scan(g2.p.lidar, fig, ax[1,2], FOV = 60, ignore_boundary = True, hitters = g2hit)
# ax[1,2].axes.set_aspect('equal')
# ax[1,2].set_xlim(-1200,1200)
# ax[1,2].set_ylim(0,800)
# #-------------------------------------------------------------



plt.pause(0.01)
plt.show()