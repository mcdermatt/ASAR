import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *
#for generating scans
from player import Player 
from game import game
from matplotlib.ticker import NullFormatter


#TODO
#	get noisy scans
#	only consider points that are hitters

high_noise = 15
low_noise = 0.1
FOV = 135 #for wall
fid = 4
minpts = 3

fig, ax = plt.subplots(nrows = 3, ncols = 3)
fig.tight_layout() #meh
fig.subplots_adjust(left = 0.01, right = 0.99, hspace = 0.01, wspace = 0.01) #slightly better

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
		ax[i,j].axes.xaxis.set_ticks([-133,133])
		ax[i,j].axes.yaxis.set_ticks([266, 533])

		#for normal wall
		ax[i,j].set_aspect('equal')
		ax[i,j].xaxis.set_major_formatter(NullFormatter())
		ax[i,j].xaxis.set_ticks_position('none')
		ax[i,j].yaxis.set_major_formatter(NullFormatter())
		ax[i,j].yaxis.set_ticks_position('none')


		#for long wall
		# if j !=2:
		# 	ax[i,j].set_aspect('equal')

		if i != 0:
			ax[i,j].grid(color=(0,0,0), linestyle='-', linewidth=0.5)

# #make simple grid for wall plots in third column --------------
# ax[0,2].plot([800,800],[0,800], linestyle = '-', color = (0,0,0,0.5))
# ax[0,2].plot([1600,1600],[0,800], linestyle = '-', color = (0,0,0,0.5))
# ax[1,2].plot([-400,-400],[0,800], linestyle = '-', color = (0,0,0,0.5))
# ax[1,2].plot([400,400],[0,800], linestyle = '-', color = (0,0,0,0.5))
# ax[2,2].plot([-400,-400],[0,800], linestyle = '-', color = (0,0,0,0.5))
# ax[2,2].plot([400,400],[0,800], linestyle = '-', color = (0,0,0,0.5))
# #--------------------------------------------------------------


# show base objects -------------------------------------------
sign_file = "assets/sign.png"
sign_img = cv2.imread(sign_file)
ax[0,0].imshow(sign_img,cmap="gray", interpolation = 'bicubic')
ax[0,0].set_xlim(0,800)
ax[0,0].set_ylim(800,0)

car_file = "assets/car.png"
car_img = cv2.imread(car_file)
ax[0,1].imshow(car_img,cmap="gray", interpolation = 'bicubic')
ax[0,1].set_xlim(0,800)
ax[0,1].set_ylim(800,0)

wall_file = "assets/wall2.png"
wall_img = cv2.imread(wall_file)
ax[0,2].imshow(wall_img,cmap="gray", interpolation = 'bicubic')
ax[0,2].set_xlim(0,800)
ax[0,2].set_ylim(800,0)
# -------------------------------------------------------------

# Generate scans ----------------------------------------------
#sign
g0 = game(fig, ax[0,0], sign_img)#cv2.flip(sign_img,0))
g0.p.FOV = np.deg2rad(FOV)
g0.p.fovfid = 100
g0.p.pos = np.array([400,800])
g0.p.heading = np.pi #0 #looking straight up
#high noise scan - do first
g0.p.noiseScale = high_noise
g0hit2 = g0.p.draw(show = False, ignore_boundary = True)
pp0_high = draw_scan(g0.p.lidar, fig, ax[2,0], FOV = FOV, ignore_boundary = True, hitters = g0hit2)
ax[2,0].axes.set_aspect('equal')
ax[2,0].set_xlim(-400,400)
ax[2,0].set_ylim(0,800)
#low noise scan - do this 2nd
g0.p.noiseScale = low_noise
g0hit = g0.p.draw(ignore_boundary = True)
pp0_low = draw_scan(g0.p.lidar, fig, ax[1,0], FOV = FOV, ignore_boundary = True, hitters = g0hit)
ax[1,0].axes.set_aspect('equal')
ax[1,0].set_xlim(-400,400)
ax[1,0].set_ylim(0,800)

#car
g1 = game(fig, ax[0,1], car_img)#cv2.flip(car_img,0))
g1.p.FOV = np.deg2rad(FOV)
g1.p.fovfid = 100
g1.p.pos = np.array([400,800])
g1.p.heading = np.pi #0 #looking straight up
#high noise scan
g1.p.noiseScale = high_noise
g1hit2 = g1.p.draw(show = False, ignore_boundary = True)
pp1_high = draw_scan(g1.p.lidar, fig, ax[2,1], FOV = FOV, ignore_boundary = True, hitters = g1hit2)
ax[2,1].axes.set_aspect('equal')
ax[2,1].set_xlim(-400,400)
ax[2,1].set_ylim(0,800)
#low noise scan
g1.p.noiseScale = low_noise
g1hit = g1.p.draw(ignore_boundary = True)
pp1_low = draw_scan(g1.p.lidar, fig, ax[1,1], FOV = FOV, ignore_boundary = True, hitters = g1hit)
ax[1,1].axes.set_aspect('equal')
ax[1,1].set_xlim(-400,400)
ax[1,1].set_ylim(0,800)
#TODO: get elements of pp1 where g1hit==1

#wall
g2 = game(fig, ax[0,2], wall_img)#cv2.flip(wall_img,0)
g2.p.FOV = np.deg2rad(FOV)
g2.p.fovfid = 100
# g2.p.pos = np.array([1200,800])
g2.p.pos = np.array([400,800])
g2.p.heading = np.pi #0 #looking straight up
#high noise scan
g2.p.noiseScale = high_noise
g2hit2 = g2.p.draw(show = False, ignore_boundary = True)
pp2_high = draw_scan(g2.p.lidar, fig, ax[2,2], FOV = FOV, ignore_boundary = True, hitters = g2hit2)
ax[2,2].axes.set_aspect('equal')
# ax[2,2].set_xlim(-1200,1200)
ax[2,2].set_xlim(-400,400)
ax[2,2].set_ylim(0,800)
#low noise scan
g2.p.noiseScale = low_noise
g2hit = g2.p.draw(ignore_boundary = True)
pp2_low = draw_scan(g2.p.lidar, fig, ax[1,2], FOV = FOV, ignore_boundary = True, hitters = g2hit)
ax[1,2].axes.set_aspect('equal')
# ax[1,2].set_xlim(-1200,1200)
ax[1,2].set_xlim(-400,400)
ax[1,2].set_ylim(0,800)

#-------------------------------------------------------------


#generate covariance ellipses --------------------------------

#sign
subdivide_scan(pp0_low[np.squeeze(np.argwhere(g0hit))], fig, ax[1,0], fidelity = fid, min_num_pts = minpts, flag = True)
subdivide_scan(pp0_high[np.squeeze(np.argwhere(g0hit2))], fig, ax[2,0], fidelity = fid, min_num_pts = minpts, flag = True)

#car
subdivide_scan(pp1_low[np.squeeze(np.argwhere(g1hit))], fig, ax[1,1], fidelity = fid, min_num_pts = minpts, flag = True )
subdivide_scan(pp1_high[np.squeeze(np.argwhere(g1hit2))], fig, ax[2,1], fidelity = fid, min_num_pts = minpts, flag = True)

#wall
subdivide_scan(pp2_low[np.squeeze(np.argwhere(g2hit))], fig, ax[1,2], fidelity = fid, min_num_pts = 10, flag = True)
subdivide_scan(pp2_high[np.squeeze(np.argwhere(g2hit2))], fig, ax[2,2], fidelity = fid, min_num_pts = 10, flag = True)

# ------------------------------------------------------------
# ax[2,2].grid(color = 'r', linestyle = '-', linewidth = 2, zorder = 1)



plt.pause(0.01)
plt.show()