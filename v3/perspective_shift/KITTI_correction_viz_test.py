from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import numpy as np
import tensorflow as tf
import time

#NOTE: make sure tf23 conda env is actiated

#init vedo and scene
plt = Plotter(N = 1, axes = 1, bg = (1, 1, 1), interactive = True) #axis = 4
disp = []

#load model
model = tf.keras.models.load_model("Net.kmod")
# model = tf.keras.models.load_model("PIPEnet10.kmod")

#read in dense point cloud of car
points_per_sample = 25 #num pts per scan - defined in MatLab script
d1 = np.loadtxt('ICET_KITTI_frame0.txt')
d2 = np.loadtxt('ICET_KITTI_frame1.txt')
scan1 = tf.reshape(tf.convert_to_tensor(d1), [-1, points_per_sample, 3])
scan2 = tf.reshape(tf.convert_to_tensor(d2), [-1, points_per_sample, 3])

#split data into training and validation sets
tsplit = 0.0 #this fraction goes into training
ntrain = int(tsplit*tf.shape(scan1)[0].numpy())
x_train = tf.concat((scan1[:ntrain], scan2[:ntrain]), axis = 1)
x_test = tf.concat((scan1[ntrain:], scan2[ntrain:]), axis = 1)

#appy model to points
n = 1 #sample number (from x_test)

c1 = np.array([x_test[n,:points_per_sample,0].numpy(), x_test[n,:points_per_sample,1].numpy(), x_test[n,:points_per_sample,2].numpy()])
c2 = np.array([x_test[n,points_per_sample:,0].numpy(), x_test[n,points_per_sample:,1].numpy(), x_test[n,points_per_sample:,2].numpy()])
# draw scans 1 and 2
disp.append(Points(c1, c = 'red', r = 10, alpha = 0.4))
disp.append(Points(c2, c = 'blue', r = 10))

inputs = x_test[n][None,:]
runlen = 5 #number of iterations to run iterative PC matching
correction = 0
for i in range(runlen):
    correction += 0.1*model.predict(inputs)[0] #show what the network thinks
#     correction = 0.1*y_test[n] #show actual solution
    c1_new = np.array([c1[0,:] + correction[0], c1[1,:] + correction[1], c1[2,:] + correction[2]])
    inputs = np.append(c1_new, c2, axis = 1).T[None,:,:]

plt.show(disp, "DNN registration test")

disp.append(Points(c1_new, c = 'red', r = 10))

print("\n estimated solution: \n :", correction)

#draw and close
plt.show(disp, "DNN Model on subsampled KITTI voxel data \n via ICET")

ViewInteractiveWidget(plt.window)