from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET

## init KITTI dataset -----------------------------------------------------------------
basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'
frame_range = range(150, 151, 1)
dataset = pykitti.raw(basedir, date, drive)
velo1 = dataset.get_velo(0) # Each scan is a Nx4 array of [x,y,z,reflectance]
c1 = velo1[:,:3]
c1 = c1[c1[:,2] > -1.5] #ignore ground plane
## ------------------------------------------------------------------------------------

#single distinct cluster
# c1 = np.random.randn(10000,3)*tf.constant([0.3,0.06,0.3]) + tf.constant([0.,4.,0.])
# c1 = np.random.randn(1000,3)*3

c2 = c1 - np.array([0.1, 0.4, 0.0])

it = ICET(cloud1 = c1, cloud2 = c2,  fid = 50)
ViewInteractiveWidget(it.plt.window)