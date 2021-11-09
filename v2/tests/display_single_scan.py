from vedo import *
import numpy as np
import pykitti
import os
from time import sleep

numEll = 0
basedir = 'C:/kitti/'
date = '2011_09_26'
drive = '0005'

c = []

plt1 = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)

frame_range = range(150, 151, 1)

dataset = pykitti.raw(basedir, date, drive)

velo1 = dataset.get_velo(0) # Each scan is a Nx4 array of [x,y,z,reflectance]
cloud1 = velo1[:,:3]
c1 = Points(cloud1, c = (0.5,0.5,1), alpha = 0.2)
c.append(c1)

velo2 = dataset.get_velo(2) # Each scan is a Nx4 array of [x,y,z,reflectance]
cloud2 = velo2[:,:3]
c2 = Points(cloud2, c = (1,0.5,0.5), alpha = 0.2)
c.append(c2)



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