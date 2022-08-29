from vedo import *
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
import numpy as np
import tensorflow as tf
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from metpy.calc import lat_lon_grid_deltas
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from ICET_spherical import ICET


#script to convert KITTI data into text files for processing in MatLab

num_frames = 155

basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'
dataset = pykitti.raw(basedir, date, drive)

for i in range(num_frames):

	print("\n ~~~~~~~~~~~~~~~~~~ Scan ",  i," ~~~~~~~~~~~~~~~~~~~~~~~~~ \n")

	velo1 = dataset.get_velo(i) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c1 = velo1[:,:3]

	fn = "C:/kitti/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data_text/scan" + str(i) + ".txt"
	np.savetxt(fn, c1)