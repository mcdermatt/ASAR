import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import *
import time
from  ICP import ICP_least_squares
from NDT import NDT

#for generating scans
from player import Player 
from game import game
from matplotlib.ticker import NullFormatter



fig, ax = plt.subplots(nrows = 2, ncols = 3) #set rows to 3 for debug
fig.subplots_adjust(left = 0.1, right = 0.9, hspace = 0.03, wspace = 0.03) #slightly better

ax[0,0].set_title("ICP")
ax[0,1].set_title("ICP point-to-plane")
ax[0,2].set_title("NDT")

for i in range(2):
	for j in range(3):
		# ax[i, j].set_aspect('equal')
		ax[i, j].xaxis.set_major_formatter(NullFormatter())
		ax[i, j].xaxis.set_ticks_position('none')
		ax[i, j].yaxis.set_major_formatter(NullFormatter())
		ax[i, j].yaxis.set_ticks_position('none')
		ax[i, j].set_xlim([-1,31])
		ax[i, j].set_ylim([-16,16])

ax[0,0].set_ylabel("Association")
ax[1,0].set_ylabel("Transformation")

#create dummy data----------------
angle = np.pi / 8
R_true = np.array([[np.cos(angle), -np.sin(angle)], 
                   [np.sin(angle),  np.cos(angle)]])
t_true = np.array([[-2], [-5]])
# Generate data as a list of 2d points
num_points = 30
true_data = np.zeros((2, num_points))
true_data[0, :] = range(0, num_points)
true_data[1, :] = 0.2 * true_data[0, :] * np.cos(0.125 * true_data[0, :]) + np.random.randn(num_points)*0.1
# Move the data
moved_data = R_true.dot(true_data) + t_true + .25*np.random.randn(num_points) #roate and add noise
moved_data = moved_data[:,8:] #cut move dataset short
# moved_data[:, 0] = np.array([-20,5]) #create outliar at start
# Assign to variables we use in formulas.
ICP_Q = true_data[:,:]
ICP_P = moved_data[:,:]
#---------------------------------
markersize = 5

#ICP (vanilla)--------------------
ICP_p1, = ax[0, 0].plot(ICP_P[0,:],ICP_P[1,:], 'g.', ms = markersize)
ICP_q1, = ax[0, 0].plot(ICP_Q[0,:], ICP_Q[1,:], 'b.', ms = markersize)

ICP_p2, t, rot = ICP_least_squares(ICP_Q,ICP_P,fig,ax[0,0], num_cycles = 1, draw = True, draw_output = False)

ax[1, 0].plot(ICP_p2[0,:],ICP_p2[1,:], 'g.', ms = markersize)
ax[1, 0].plot(ICP_Q[0,:], ICP_Q[1,:], 'b.', ms = markersize)
#---------------------------------

#ICP point-to-plane---------------------------------------------------------------
ICP_p2p_p1, = ax[0, 1].plot(ICP_P[0,:],ICP_P[1,:], 'g.', ms = markersize)
ICP_p2p_q1, = ax[0, 1].plot(ICP_Q[0,:], ICP_Q[1,:], 'b.', ms = markersize)

#subdivide base scan into 3 "planes"
plane1 = true_data[:, :num_points//3]
plane2 = true_data[:, num_points//3:2*num_points//3]
plane3 = true_data[:, 2*num_points//3:]

#generate a best fit line to represent each plane
bf1 = np.polyfit(plane1[0,:], plane1[1,:], 1) #y = mx + b
#l1(x) places point correctly on y coordinate of plane 1 line
l1 = lambda x: x*bf1[0] + bf1[1]
line1, = ax[0,1].plot(plane1[0,:], l1(plane1[0,:]), '-', color = (0,0,0,0.5) )

bf2 = np.polyfit(plane2[0,:], plane2[1,:], 1) #y = mx + b
l2 = lambda x: x*bf2[0] + bf2[1]
line2, = ax[0,1].plot(plane2[0,:], l2(plane2[0,:]), '-', color = (0,0,0,0.5) )

bf3 = np.polyfit(plane3[0,:], plane3[1,:], 1) #y = mx + b
l3 = lambda x: x*bf3[0] + bf3[1]
line3, = ax[0,1].plot(plane3[0,:], l3(plane3[0,:]), '-', color = (0,0,0,0.5) )

on_line = np.zeros([2, max(np.shape(ICP_P))])
for i in range(max(np.shape(ICP_P))):
	#get correspondences- draw line between points in pp2 and closest corresponding points on best fit lines
	pt = ICP_P[:,i] #arbitrary test point (TODO: put in loop)

	if pt[0] < plane1[0,-1]:
		#gets y intercetp for line passing through desired point
		# y = mx + b -> b = y - mx
		get_b = lambda pt: pt[1] + (1/bf1[0])*pt[0]
		b = get_b(pt)
		x = (b - bf1[1])/(bf1[0]+ (1/bf1[0]))
		ax[0,1].plot([pt[0],x],[pt[1], l1(x)], '--', color =(0,0,0,1), lw = 1)
		on_line[:,i] = [x, l1(x)]

	if pt[0] > plane1[0,-1] and pt[0] < plane2[0,-1] :
		#gets y intercetp for line passing through desired point
		# y = mx + b -> b = y - mx
		get_b = lambda pt: pt[1] + (1/bf2[0])*pt[0]
		b = get_b(pt)
		x = (b - bf2[1])/(bf2[0]+ (1/bf2[0]))
		ax[0,1].plot([pt[0],x],[pt[1], l2(x)], '--', color =(0,0,0,1), lw = 1)
		on_line[:,i] = [x, l2(x)]


	if pt[0] > plane2[0,-1] :
		#gets y intercetp for line passing through desired point
		# y = mx + b -> b = y - mx
		get_b = lambda pt: pt[1] + (1/bf3[0])*pt[0]
		b = get_b(pt)
		x = (b - bf3[1])/(bf3[0]+ (1/bf3[0]))
		ax[0,1].plot([pt[0],x],[pt[1], l3(x)], '--', color =(0,0,0,1), lw = 1)
		on_line[:,i] = [x, l3(x)]


	#add to array containing all closest points on best fit line
# print(on_line)
#just doing least squares ICP on points projected on best fit planes
ICP_p2p, t, rot = ICP_least_squares(on_line,ICP_P,fig,ax[0,1], num_cycles = 1, draw = False, draw_output = False)
# print(t, rot)

ax[1, 1].plot(ICP_p2p[0,:],ICP_p2[1,:], 'g.', ms = markersize)
ax[1, 1].plot(ICP_Q[0,:], ICP_Q[1,:], 'b.', ms = markersize)

#---------------------------------------------------------------------------

#NDT------------------------------

NDT_p1, = ax[0, 2].plot(moved_data[0,:],moved_data[1,:], 'g.', ms = markersize)
NDT_q1, = ax[0, 2].plot(true_data[0,:], true_data[1,:], 'b.', ms = markersize)
# print(np.shape(true_data))
minx = np.min(true_data[0,:])
maxx = np.max(true_data[0,:])
miny = np.min(true_data[1,:])
maxy = np.max(true_data[1,:])
lims = np.array([minx, maxx, miny, maxy])
# print(lims)
r, t, results = NDT(true_data[:,:],moved_data[:,:],fig,ax[0,2], fid = 4, 
                              num_cycles = 100, along_track_demo = 'generate_graphic', lims = lims, draw_output = False)
# print(r, t)
P_corrected = moved_data.T.dot(R(-r)) + t.T 
P_corrected = P_corrected.T
# NDT_p1, = ax[1, 2].plot(moved_data[0,:],moved_data[1,:], 'g.', ms = markersize)
NDT_p2 = ax[1, 2].plot(P_corrected[0,:],P_corrected[1,:], 'g.', ms = markersize)
NDT_q2 = NDT_q1, = ax[1, 2].plot(true_data[0,:], true_data[1,:], 'b.', ms = markersize)

# ax[2,2].plot(results) #for debug

#---------------------------------

plt.pause(0.01)
plt.show()