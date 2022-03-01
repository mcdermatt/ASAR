from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
from scene import scene
import numpy as np

## init KITTI dataset -----------------------------------------------------------------
basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'
frame_range = range(150, 151, 1)
dataset = pykitti.raw(basedir, date, drive)
velo1 = dataset.get_velo(0) # Each scan is a Nx4 array of [x,y,z,reflectance]
cloud = velo1[:,:3]
cloud = cloud[cloud[:,2] > -1.5] #ignore ground plane
## ------------------------------------------------------------------------------------

#basic cartesian
# S = scene(cloud = cloud, fid = 40, cull = False)

#spherical 
# cloud = np.array([[10,10,-1]])
S2 = scene(cloud = cloud, fid = 40, cull = True, coord = 1)