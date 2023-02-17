from vedo import *
import numpy as np
import tensorflow as tf
import trimesh
import time
import pickle


#limit GPU memory ------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    memlim = 4*1024
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memlim)])
  except RuntimeError as e:
    print(e)
#-----------------------------------------------------------------

from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf
import os
from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET

from pioneer.das.api.platform import Platform
from scipy.spatial.transform import Rotation as R


# # init params -------------------------------------------------------------------------------
# drive = "20200721_144638_part36_1956_2229" #big hill and church:
# num_frames = 274

# drive = "20200617_191627_part12_1614_1842"#straight road, narrow with pedestrians and shops 
# num_frames = 230

# drive = "20200706_161206_part22_670_950" #suburban neighborhood, trees, houses and parked cars
# num_frames = 274

# drive = "20200721_154835_part37_696_813" #overgrown highway
# num_frames = 120

# drive = "20200618_191030_part17_1120_1509" #long straight stretch, passing cyclists
# num_frames = 388

# drive = "20200706_191736_part30_1721_1857" #big city, jaywalker in flip flops
# num_frames = 136

# drive = "20200805_002607_part48_2083_2282" #boulevard on rainy night
# num_frames = 200

# drive = "20200706_202209_part31_2980_3091" #straight drive, one way road, urban, big trees
# num_frames = 150

drive = "20200803_151243_part45_4780_5005" #tight alleyway in the rain, 235 frames
num_frames = 230

numShifts = 5 #number of times to resample and translate each voxel each scan
npts = 100
skips = 3 #num of scans to skip between each frame
#----------------------------------------------------------------------------------------------



#extract ground truth vehicle motion from GNSS/ INS -------------------------------------------
dataset_path = "/media/derm/06EF-127D3/leddartech/" + drive
config_path = "/media/derm/06EF-127D3/leddartech/" + drive + "/platform.yml"
pf = Platform(dataset_path, config_path)

GNSS = pf.sensors['sbgekinox_bcc']


from pioneer.das.api.egomotion.imu_egomotion_provider import IMUEgomotionProvider as emp 
name = pf
test = emp(name, GNSS['navposvel'], GNSS['ekfeuler'])
timestamps = test.get_timestamps()
print(len(timestamps))
T1 = test.get_transform(timestamps[1])
T2 = test.get_transform(timestamps[2])

gt_vec = np.zeros([len(timestamps)-1,6])

for i in range(1,len(timestamps)-skips):
    #get translations from GNSS/INS baseline
    gt_vec[i-1,0] = test.get_transform(timestamps[i+skips])[1,3] - test.get_transform(timestamps[i])[1,3]
    gt_vec[i-1,1] = test.get_transform(timestamps[i+skips])[0,3] - test.get_transform(timestamps[i])[0,3]
    gt_vec[i-1,2] = test.get_transform(timestamps[i+skips])[2,3] - test.get_transform(timestamps[i])[2,3]
    #get rotations
    T1 = test.get_transform(timestamps[i])
    T2 = test.get_transform(timestamps[i+skips])
    r1 = R.from_matrix(T1[:3,:3])
    r2 = R.from_matrix(T2[:3,:3])
    gt_vec[i-1,3:] = (r2.as_euler('xyz', degrees=False) - r1.as_euler('xyz', degrees=False))
    
vf = np.sqrt(gt_vec[:,0]**2 + gt_vec[:,1]**2)
gt_vec[:,0] = vf
gt_vec[:,1] = 0
gt_vec[:,2] = 0
gt_vec = gt_vec * 5

#----------------------------------------------------------------------------------------------

# MAIN LOOP
for idx in range(num_frames):
  print("\n ~~~~~~~~~ Frame #", idx, "~~~~~~~~~~~~~ \n")

  #get unidstorted point clouds ----------------

  data1 = pf['ouster64_bfc_xyzit'][idx].get_point_cloud(undistort = True)
  data2 = pf['ouster64_bfc_xyzit'][idx+skips].get_point_cloud(undistort = True)
  ts_lidar = pf['ouster64_bfc_xyzit'][idx].timestamp

  # #get point clouds - old distorted way-------
  # prefix = "/media/derm/06EF-127D2/leddartech/"+ drive + "/ouster64_bfc_xyzit/"
  # fn1 = prefix + '%08d.pkl' %(idx)
  # with open(fn1, 'rb') as f:
  #     data1 = pickle.load(f)
  # ts_lidar = np.asarray(data1.tolist())[0,-1]
  # #[x, y, z, intensity, timestamp]
  # data1 = np.asarray(data1.tolist())[:,:3]
  # fn2 = prefix + '%08d.pkl' %(idx+1)
  # with open(fn2, 'rb') as f:
  #     data2 = pickle.load(f)
  # data2 = np.asarray(data2.tolist())[:,:3]
  # #--------------------------------------------

  #get ground truth from GNSS data
  # loop through all GNSS timestamps, stop when larger than ts_lidar and use previous index
  for c in range(len(timestamps)):
      ts_gnss = timestamps[c]
      if ts_gnss > ts_lidar:
          break
  x0 = tf.convert_to_tensor(gt_vec[c], dtype = tf.float32)
  # print(x0)

  shift_scale = 0.0 #standard deviation by which to shift the grid BEFORE SAMPLING corresponding segments of the point cloud
  shift = tf.cast(tf.constant([shift_scale*tf.random.normal([1]).numpy()[0], shift_scale*tf.random.normal([1]).numpy()[0], 0.2*shift_scale*tf.random.normal([1]).numpy()[0], 0, 0, 0]), tf.float32)

  data1 = data1[data1[:,2] > -0.75] #ignore ground plane
  data2 = data2[data2[:,2] > -0.75] #ignore ground plane

  it = ICET(cloud1 = data1, cloud2 = data2, fid = 50, niter = 2, draw = False, group = 2, 
    RM = True, DNN_filter = False, cheat = x0+shift)

  #Get ragged tensor containing all points from each scan inside each sufficient voxel
  in1 = it.inside1
  npts1 = it.npts1
  in2 = it.inside2
  npts2 = it.npts2
  corr = it.corr #indices of bins that have enough points from scan1 and scan2

  #get indices of rag with enough elements
  ncells = tf.shape(corr)[0].numpy() #num of voxels with sufficent number of points
  # print(tf.gather(npts2, corr))
  enough1 = tf.gather(in1, corr)
  enough2 = tf.gather(in2, corr)

  for j in range(numShifts):
    #init array to store indices
    idx1 = np.zeros([ncells ,npts])
    idx2 = np.zeros([ncells ,npts])

    #loop through each element of ragged tensor
    for i in range(ncells):
        idx1[i,:] = tf.random.shuffle(enough1[i])[:npts].numpy() #shuffle order and take first 25 elements
        idx2[i,:] = tf.random.shuffle(enough2[i])[:npts].numpy() #shuffle order and take first 25 elements

    idx1 = tf.cast(tf.convert_to_tensor(idx1), tf.int32) #indices in scan 1 of points we've selected
    idx2 = tf.cast(tf.convert_to_tensor(idx2), tf.int32) 

    from1 = tf.gather(it.cloud1_tensor, idx1)
    # from2 = tf.gather(it.cloud2_tensor_OG, idx2) #corresponding points in OG pose
    from2 = tf.gather(it.cloud2_tensor, idx2)      #transformed by "ground truth" translation

    scan1 = tf.reshape(from1, [-1, 3]).numpy()
    scan2 = tf.reshape(from2, [-1, 3]).numpy()

    #randomly translate each sample from scan 2
    rand = tf.constant([1., 1., 0.1])*tf.random.normal([ncells, 3])
    #tile and apply to scan2
    t = tf.tile(rand, [npts,1])
    t = tf.reshape(tf.transpose(t), [3,npts,-1])
    t = tf.transpose(t, [2,1,0])
    t = tf.reshape(t, [-1, 3])
    scan2 += t.numpy()

    full_soln_vec = rand + shift[:3]
    compact_soln_vec = it.L @ tf.transpose(it.U, [0,2,1]) @ full_soln_vec[:,:,None] #remove extended axis
    compact_soln_vec = tf.matmul(it.U, compact_soln_vec) #project back to XYZ
    compact_soln_vec = compact_soln_vec[:,:,0] #get rid of extra dimension

    soln = full_soln_vec #consider entire solution vector (compact and extended directions)
    # soln = compact_soln_vec #only consider ground truth solutions in directions deemed useful by ICET

    #initialize array to store data on first iteration
    if idx*(j+1) == 0:
      scan1_cum = scan1
      scan2_cum = scan2
      soln_cum = rand + shift[:3]
    else:
      scan1_cum = np.append(scan1_cum, scan1, axis = 0)
      scan2_cum = np.append(scan2_cum, scan2, axis = 0)
      soln_cum = np.append(soln_cum, soln, axis = 0)

  print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

  #periodically save so we don't lose everything...
  if i % 10 == 0:
    print("saving...")
    np.save('/media/derm/06EF-127D3/TrainingData/leddartech/rainyAlleySkip3_scan1_100pts', scan1_cum)
    np.save('/media/derm/06EF-127D3/TrainingData/leddartech/rainyAlleySkip3_scan2_100pts', scan2_cum)
    np.save('/media/derm/06EF-127D3/TrainingData/leddartech/rainyAlleySkip3_ground_truth_100pts', soln_cum)



np.save('/media/derm/06EF-127D3/TrainingData/leddartech/rainyAlleySkip3_scan1_100pts', scan1_cum)
np.save('/media/derm/06EF-127D3/TrainingData/leddartech/rainyAlleySkip3_scan2_100pts', scan2_cum)
np.save('/media/derm/06EF-127D3/TrainingData/leddartech/rainyAlleySkip3_ground_truth_100pts', soln_cum)



