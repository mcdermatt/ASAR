from vedo import *
import numpy as np
import pykitti
import os
from time import sleep

numEll = 10
basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'

plt1 = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)

frame_range = range(150, 151, 1)

dataset = pykitti.raw(basedir, date, drive)

velo = dataset.get_velo(0) # Each scan is a Nx4 array of [x,y,z,reflectance]
cloud = velo[:,:3]
print(np.shape(cloud))

#pure point cloud
c = Points(cloud, c = (1,1,1), alpha = 0.2)
# plt1.show(c, "Cloud", at=0)

#opens parallel window to display 
# cloud = cloud[:10000]
# surf = recoSurface(cloud)
# plt1.show(surf, at=1, axes=7, zoom=1.2, interactive=1).close()

#generate a bunch of ellipses
E = []
for i in range(numEll):
	ell = Ellipsoid(pos=(np.random.randn()*10, np.random.randn()*10, 
		np.random.rand()*5), axis1=(1, 0, 0), axis2=(0, 2, 0), axis3=(np.random.rand(), np.random.rand(), np.random.rand()), 
		c=(np.random.rand(), np.random.rand(), np.random.rand()), alpha=1, res=12)

	E.append(ell)

plt1.show(c,E, "Kitti Data Test", at =0).close()