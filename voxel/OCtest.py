from vedo import *
# settings.embedWindow(backend='ipyvtk', verbose = True)
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
# cloud = cloud[cloud[:,2] > -1.5] #ignore ground plane
## ------------------------------------------------------------------------------------


S = scene(cloud = cloud, fid = 20)

ViewInteractiveWidget(S.plt.window)