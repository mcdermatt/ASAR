from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import numpy as np
import tensorflow as tf
import time

#NOTE: MAKE SURE PROPER CONDA ENV IS ACTIVATED TO RUN THIS!!!

#init vedo and scene
plt = Plotter(N = 1, axes = 1, bg = (1, 1, 1), interactive = True) #axis = 4
disp = []

#load model
model = tf.keras.models.load_model("Net.kmod")

#read in dense point cloud of car
points_per_sample = 25 #num pts per scan - defined in MatLab script
# d1 = np.loadtxt('training_data/car_demo_scan1.txt')
# d2 = np.loadtxt('training_data/car_demo_scan2.txt')
# gt = np.loadtxt('training_data/car_demo_ground_truth.txt')
d1 = np.loadtxt('viz_scan1.txt')
d2 = np.loadtxt('viz_scan2.txt')
gt = np.loadtxt('viz_ground_truth.txt')
true_pos1 = np.loadtxt('viz_true_pos1.txt')
scan1 = tf.reshape(tf.convert_to_tensor(d1), [-1, points_per_sample, 3])
scan2 = tf.reshape(tf.convert_to_tensor(d2), [-1, points_per_sample, 3])
gt = tf.convert_to_tensor(gt)

#split data into training and validation sets
tsplit = 0.0 #this fraction goes into training
ntrain = int(tsplit*tf.shape(scan1)[0].numpy())
x_train = tf.concat((scan1[:ntrain], scan2[:ntrain]), axis = 1)
x_test = tf.concat((scan1[ntrain:], scan2[ntrain:]), axis = 1)
y_train = gt[:ntrain]
y_test = gt[ntrain:]
# print(tf.shape(x_train))

#appy model to points
n = 1 #sample number (from x_test)
# print(tf.shape(x_test))

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
print("\n correct solution: \n :", y_test[n]*0.1)
print("\n error: \n :", y_test[n]*0.1 - correction, "meters")


#display mesh of dummy pedestrian for reference
fname = 'viz_model.stl'
# ped1 = Mesh(fname).c("gray").alpha(0.5).rotate(true_pos1[n,3], axis = (0,0,1))
# ped1.pos(true_pos1[n,0], true_pos1[n,1], -1.72 + true_pos1[n,2])
# disp.append(ped1)

plt.show(disp, "DNN registration test")

ped2 = Mesh(fname).c("gray").alpha(0.5)#.rotate(true_pos1[n,3], axis = (0,0,1))
ped2.pos(true_pos1[n,0] + y_test[n,0]*0.1, true_pos1[n,1] + y_test[n,1]*0.1, -1.72 + true_pos1[n,2] + y_test[n,2]*0.1)
disp.append(ped2)


#test human COM
# disp.append(Point(true_pos1[n,:3], c = 'yellow', r = 30))

#draw and close
plt.show(disp, "DNN registration test")
# disp = []
# disp.append(Points(c2, c = 'blue', r = 10))
# disp.append(Points(c1_new, c = 'red', r = 10))
# plt.show(disp, "DNN registration test")


ViewInteractiveWidget(plt.window)