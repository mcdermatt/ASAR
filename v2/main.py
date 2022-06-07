import numpy as np
import tensorflow as tf
#need to have these two lines to work on my ancient 1060 3gb
#  https://stackoverflow.com/questions/43990046/tensorflow-blas-gemm-launch-failed
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
from utils import *
import tensorflow_probability as tfp
import time
import os
from ipyvtklink.viewer import ViewInteractiveWidget
import pykitti
from ICET3D import ICET3D

"""Runs the same script as the ICET3D notebook but with better performance due to not using inline display"""

#NOTE: Out of Memory Error comes from too high fidelity/ pts in cloud tensor --> 100x100x2x120,000 > 2gb


nc = 3	 #number of cycles
mnp = 10 #50#100 #minimum number of points per voxel
D = True #draw sim
DG = False #draw grid
DE = True #draw ellipsoids
DC = True #draw correspondences
TD = False #use test dataset
CM = "voxel" #correspondence method, "voxel" or "NN"
vizL = True #draw arrows in direction of non-truncated directions for each distribution

# plt = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)
plt = Plotter(N=1, axes=4, interactive=True)
# plt = Plotter(N=1, axes=4, resetcam=False)


#copy and paste from terminal using shift+c on Vedo window-----------------------
plt.camera.SetPosition( [-1.435, 1.429, 54.846] )
plt.camera.SetFocalPoint( [0., 0., 0.154] )
plt.camera.SetViewUp( [0.0, -1.0, 0.0] )
plt.camera.SetDistance( 54.764 )
plt.camera.SetClippingRange( [48.342, 61.103] )
plt.camera.SetClippingRange( [0.154, 154.418] )
#--------------------------------------------------------------------------------

## Use real data ----------------------------------------------------------------
# basedir = 'C:/kitti/'
# date = '2011_09_26'
# drive = '0005'
# frame_range = range(150, 151, 1)
# dataset = pykitti.raw(basedir, date, drive)
# velo1 = dataset.get_velo(0) # Each scan is a Nx4 array of [x,y,z,reflectance]
# cloud1 = velo1[:,:3]
# cloud1_tensor = tf.convert_to_tensor(cloud1, np.float32)
# velo2 = dataset.get_velo(2) # Each scan is a Nx4 array of [x,y,z,reflectance]
# cloud2 = velo2[:,:3]
# cloud2_tensor = tf.convert_to_tensor(cloud2, np.float32)
# ---------------------------------------------------------------------------------

start = time.time()

# #use whole point set
# #---------------------------------------------------------------------------------
# f = tf.constant([50,50,2]) #fidelity in x, y, z # < 5s
# lim = tf.constant([-100.,100.,-100.,100.,-10.,10.]) #needs to encompass every point
# npts = 100000
# Q, x_hist = ICET3D(cloud1_tensor[:npts], cloud2_tensor[:npts], plt, bounds = lim, 
#            fid = f, num_cycles = nc , min_num_pts = mnp, draw = D, draw_grid = DG, 
#            draw_ell = DE, draw_corr = DC)

# #---------------------------------------------------------------------------------

#just consider small section of image where there are easily identifiable features:
#----------------------------------------------------------------------------------
# limtest = tf.constant([-20.,20.,-20.,20.,-1.5,1.5])
# # f = tf.constant([35,35,35])
# # f = tf.constant([9,30,1]) #for 2d paper viz
# # f = tf.constant([17,17,17])
# f = tf.constant([20,13,20])

## load custom point cloud geneated in matlab------------------------------------------
# cloud1 = np.loadtxt(r"C:\Users\Derm\vaRLnt\v3\scene1_scan1_thick.txt", dtype = float)
# cloud2 = np.loadtxt(r"C:\Users\Derm\vaRLnt\v3\scene1_scan2_thick.txt", dtype = float)

# cloud1 = np.loadtxt(r"C:\Users\Derm\vaRLnt\v3\mountain_scan1_no_trees.txt", dtype = float)
# cloud2 = np.loadtxt(r"C:\Users\Derm\vaRLnt\v3\mountain_scan2_no_trees.txt", dtype = float)

cloud1 = np.loadtxt(r"C:\Users\Derm\vaRLnt\v3\MC_trajectories\scene1_scan17.txt", dtype = float)
cloud2 = np.loadtxt(r"C:\Users\Derm\vaRLnt\v3\MC_trajectories\scene1_scan18.txt", dtype = float)


c1 = cloud1 + 0.02*np.random.randn(np.shape(cloud1)[0], 3)
c2 = cloud2 + 0.02*np.random.randn(np.shape(cloud2)[0], 3)


# if scene == 1 ----------
limtest = tf.constant([-60., 60., -20., 20., -5.,5.])
f = tf.constant([18,6,3]) # was this for MC sim
# f = tf.constant([30,10,10]) #test
#Problem seems to be points that are on outer rings of scan
#  -> need to remove any points in scan1 > 25m(?) from center 
c1 = c1[c1[:,0] > -30]
c1 = c1[c1[:,0] < 30]
# c1 = c1[c1[:,2] > -1.25] #ignore ground plane
# c2 = c2[c2[:,2] > -1.25] #ignore ground plane
#-------------------------

# # # if scene == 2 --------
# limtest = tf.constant([-100., 100., -100., 100., -30.,30.])
# f = tf.constant([40,40,1]) 
# #-------------------------


#rotate scans
rot = R_tf(tf.constant([0., 0., 0.05]))
c2 = c2 @ rot.numpy() 
# c2 = c2 + tf.constant([0.5, 0., 0.]) #test - give it a hint to get things started

cloud1_tensor = tf.convert_to_tensor(c1, dtype = tf.float32)
cloud2_tensor = tf.convert_to_tensor(c2, dtype = tf.float32)
## ------------------------------------------------------------------------------------

# cloud1_tensor = None
# cloud2_tensor = None


Q, x_hist = ICET3D(cloud1_tensor, cloud2_tensor, plt, bounds = limtest, 
           fid = f, num_cycles = nc , min_num_pts = mnp, draw = D, draw_grid = DG,
           draw_ell = DE, draw_corr = DC, test_dataset = TD, CM = CM, vizL = vizL)

print("pred stds:", np.sqrt(np.abs(np.diag(Q))))

#----------------------------------------------------------------------------------


print("took", time.time() - start, "seconds total")