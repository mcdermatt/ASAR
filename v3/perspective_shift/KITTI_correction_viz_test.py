from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import numpy as np
import tensorflow as tf
import time

#NOTE: make sure tf23 conda env is actiated

n = 188 #187 #110 #sample number (from x_test)

#init vedo and scene
plt = Plotter(N = 1, axes = 1, bg = (1, 1, 1), interactive = True) #axis = 4
disp = []

#load model
model = tf.keras.models.load_model("Net.kmod")
# model = tf.keras.models.load_model("PIPEnet10.kmod")
# model = tf.keras.models.load_model("KITTInet.kmod") #25 pts, trained well on small KITTI city
# model = tf.keras.models.load_model("FULL_KITTInet.kmod") #25 pts large KITTI 00
# model = tf.keras.models.load_model("KITTInet50.kmod")


#read in dense point cloud of car
points_per_sample = 50 #50  #num pts per scan - defined in MatLab script
# d1 = np.loadtxt('training_data/ICET_KITTI_scan1.txt')
# d2 = np.loadtxt('training_data/ICET_KITTI_scan2.txt')
#old
d1 = np.loadtxt('training_data/ICET_KITTI_scan1_50.txt')
d2 = np.loadtxt('training_data/ICET_KITTI_scan2_50.txt')
gt = np.loadtxt('training_data/ICET_KITTI_ground_truth_50.txt')

#new shifted dataset
# d1 = np.loadtxt('training_data/ICET_KITTI_FULL_scan1_to10.txt')
# d2 = np.loadtxt('training_data/ICET_KITTI_FULL_scan2_to10.txt')
# gt = np.loadtxt('training_data/ICET_KITTI_FULL_ground_truth_to10.txt')
# d1 = np.load("C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_scan1_to400_noGP.npy")
# d2 = np.load("C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_scan2_to400_noGP.npy")
# gt = np.load("C:/Users/Derm/Desktop/big/pshift/ICET_KITTI_FULL_ground_truth_to400_noGP.npy")

scan1 = tf.reshape(tf.convert_to_tensor(d1), [-1, points_per_sample, 3])
scan2 = tf.reshape(tf.convert_to_tensor(d2), [-1, points_per_sample, 3])

#split data into training and validation sets
tsplit = 0.95 #this fraction goes into training
# tsplit = 0.1
ntrain = int(tsplit*tf.shape(scan1)[0].numpy())
x_train = tf.concat((scan1[:ntrain], scan2[:ntrain]), axis = 1)
x_test = tf.concat((scan1[ntrain:], scan2[ntrain:]), axis = 1)
y_test = gt[ntrain:]

#appy model to points
c1 = np.array([x_test[n,:points_per_sample,0].numpy(), x_test[n,:points_per_sample,1].numpy(), x_test[n,:points_per_sample,2].numpy()])
c2 = np.array([x_test[n,points_per_sample:,0].numpy(), x_test[n,points_per_sample:,1].numpy(), x_test[n,points_per_sample:,2].numpy()])
# draw scans 1 and 2
disp.append(Points(c1, c = 'red', r = 10, alpha = 1))
disp.append(Points(c2, c = 'blue', r = 10, alpha = 0.4))

inputs = x_test[n][None,:]
runlen = 3 #number of iterations to run iterative PC matching
correction = 0
for i in range(runlen):
    # correction -= model.predict(inputs)[0] #show what the network thinks (was this)
    correction -= 0.1*model.predict(inputs)[0] #test
    # correction = -y_test[n] #show actual solution
    c2_new = np.array([c2[0,:] + correction[0], c2[1,:] + correction[1], c2[2,:] + correction[2]])
    inputs = np.append(c1, c2_new, axis = 1).T[None,:,:]

print("\n estimated solution: ", correction)
print("\n ground truth solution: ", y_test[n])

plt.show(disp, "DNN registration test")

disp.append(Points(c2_new, c = 'blue', r = 10))

#draw and close
plt.show(disp, "DNN Model on subsampled KITTI voxel data \n via ICET")

ViewInteractiveWidget(plt.window)