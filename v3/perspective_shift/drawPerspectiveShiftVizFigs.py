#setup - rememeber to switch to tensorflow 2.3 kernel...
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import datetime
import trimesh
import time
from vedo import *
from ipyvtklink.viewer import ViewInteractiveWidget

#need to have these two lines to work on my ancient 1060 3gb
#  https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
# physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


plt = Plotter(N = 3, axes = 4, bg = (1, 1, 1), interactive = True)
disp1 = [] #before estimated transformation (drawn on left)
disp2 = [] #after 1 transformation (drawn in center)
disp3 = [] #after niter transformations

alph = 0.8
rad = 2

#read in dense point cloud of car
points_per_sample = 350 #num pts per scan - defined in MatLab script
c1 = np.loadtxt('training_data/car_demo2_scan1.txt')
c2 = np.loadtxt('training_data/car_demo2_scan2.txt')
gt = np.loadtxt('training_data/car_demo2_ground_truth.txt')

#raw points
disp1.append(Points(c1, c = 'red', r = rad, alpha = alph))
disp1.append(Points(c2, c = 'blue', r = rad, alpha = alph))

#match cloud means
mean1 = np.mean(c1, axis = 0)
mean2 = np.mean(c2, axis = 0)
disp2.append(Points(c1, c = 'red', r = rad, alpha = alph))
disp2.append(Points(c2 + mean1 - mean2 , c = 'blue', r = rad, alpha = alph))

#draw true soln
disp3.append(Points(c1, c = 'red', r = rad, alpha = alph))
disp3.append(Points(c2 - gt/10, c = 'blue', r = rad, alpha = alph))

plt.show(disp1, "Initial offset", at = 0)
plt.show(disp2, "Matching Point Cloud Means", at = 1)
plt.show(disp3, "Correct Translation", at = 2)