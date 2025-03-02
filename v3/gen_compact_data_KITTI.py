import pykitti
import numpy as np
import tensorflow as tf
import trimesh

#limit GPU memory ------------------------------------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
  except RuntimeError as e:
    print(e)
#-----------------------------------------------------------------

from tensorflow.math import sin, cos, tan
import tensorflow_probability as tfp
from ICET_spherical import ICET
from utils import R_tf
from metpy.calc import lat_lon_grid_deltas


numShifts = 5 #5 #number of times to resample and translate each voxel each scan
noise_scale = 0.0 #add gaussian noise to each point
start_idx = 350
runLen = 300
npts = 100 #50 
skip3 = True 

# # init KITTI dataset-----------------------------------------
# basedir = 'C:/kitti/' #windows
basedir = '/media/derm/06EF-127D1/KITTI'
date = '2011_09_26'
# drive = '0005'# urban dataset used in 3D-ICET paper 
# drive = '0091'# urban dataset, 341 frames, shopping center, some pedestians (using up to 250 for train, 250+ for test)
# drive = '0095'# urban dataset, 267 frames, tight road, minimal other vehicles #start 250
drive = '0117' #657 frames through wooded campus
dataset = pykitti.raw(basedir, date, drive)
# --------------------------------------------------------------------------

# #init KC -----------------------------------------------------
# fpl = np.loadtxt("/home/derm/KITTICARLA/dataset/Town03/generated/full_poses_lidar.txt") #full poses lidar
# pl = "/home/derm/KITTICARLA/dataset/Town03/generated/poses_lidar.ply"
# datposes = trimesh.load(pl)
# true_traj = datposes.vertices
# #create rotation and translation vectors
# R = np.array([[fpl[:,0], fpl[:,1], fpl[:,2]],
#               [fpl[:,4], fpl[:,5], fpl[:,6]],
#               [fpl[:,8], fpl[:,9], fpl[:,10]]]).T

# T = np.array([fpl[:,3], fpl[:,7], fpl[:,11]]).T
# vel = np.diff(T.T)
# # --------------------------------------------------------------------------


for idx in range(start_idx, start_idx+runLen):
	print("\n ~~~~~~~~~ Frame #", idx, "~~~~~~~~~~~~~ \n")


	# KITTI --------------------------------------------------------------------------
	velo1 = dataset.get_velo(idx) # Each scan is a Nx4 array of [x,y,z,reflectance]
	c1 = velo1[:,:3]
	if skip3 == True:
		velo2 = dataset.get_velo(idx+3) # Each scan is a Nx4 array of [x,y,z,reflectance]
	else:
		velo2 = dataset.get_velo(idx+1)
	c2 = velo2[:,:3]
	c1 = c1[c1[:,2] > -1.5] #ignore ground plane
	c2 = c2[c2[:,2] > -1.5] #ignore ground plane
	# c1 = c1[c1[:,2] > -2.] #ignore reflections
	# c2 = c2[c2[:,2] > -2.] #ignore reflections

	poses0 = dataset.oxts[idx] #<- ID of 1st scan
	poses1 = dataset.oxts[idx+1] #<- ID of 2nd scan
	dt = 0.1037 #mean time between lidar samples

	if skip3 == False:
		OXTS_ground_truth = tf.constant([poses1.packet.vf*dt, -poses1.packet.vl*dt, poses1.packet.vu*dt, poses1.packet.wf*dt, poses1.packet.wl*dt, poses1.packet.wu*dt])
	else:
		poses2 = dataset.oxts[idx+2]
		poses3 = dataset.oxts[idx+3]
		gt1 = tf.constant([poses1.packet.vf*dt, -poses1.packet.vl*dt, poses1.packet.vu*dt, poses1.packet.wf*dt, poses1.packet.wl*dt, poses1.packet.wu*dt])
		gt2 = tf.constant([poses2.packet.vf*dt, -poses2.packet.vl*dt, poses2.packet.vu*dt, poses2.packet.wf*dt, poses2.packet.wl*dt, poses2.packet.wu*dt])
		gt3 = tf.constant([poses3.packet.vf*dt, -poses3.packet.vl*dt, poses3.packet.vu*dt, poses3.packet.wf*dt, poses3.packet.wl*dt, poses3.packet.wu*dt])
		OXTS_ground_truth = gt1 + gt2 + gt3
	shift_scale = 0.0 #standard deviation by which to shift the grid BEFORE SAMPLING corresponding segments of the point cloud
	shift = tf.cast(tf.constant([shift_scale*tf.random.normal([1]).numpy()[0], shift_scale*tf.random.normal([1]).numpy()[0], 0.2*shift_scale*tf.random.normal([1]).numpy()[0], 0, 0, 0]), tf.float32)

	it = ICET(cloud1 = c1, cloud2 = c2, fid = 50, niter = 3, draw = False, group = 2, 
				RM = True, DNN_filter = False, cheat = OXTS_ground_truth+shift)
	# --------------------------------------------------------------------------

	# # KITTI-CARLA ---------------------------------------------------------------------
	# skip = int(3*np.random.randn())
	# print(skip)

	# s1_fn = '/home/derm/KITTICARLA/dataset/Town03/generated/frames/frame_%04d.ply' %(idx)
	# s2_fn = '/home/derm/KITTICARLA/dataset/Town03/generated/frames/frame_%04d.ply' %(idx + skip)

	# dat1 = trimesh.load(s1_fn)
	# dat2 = trimesh.load(s2_fn)

	# c1 = dat1.vertices
	# c1 = c1[c1[:,2] > -1.5]
	# c1 = c1.dot(R[(idx)*100])
	# c1 += noise_scale*np.random.randn(np.shape(c1)[0],3)

	# c2 = dat2.vertices
	# c2 = c2[c2[:,2] > -1.5]
	# c2 = c2.dot(R[(idx+skip)*100])
	# # c2 = c2 + vel[:,(idx+skip)*100]*100*skip #transform c2 to overlay with c1
	# # c2 = c2 + (vel[:,(idx+skip)*100] + vel[:,(idx)*100])*50*skip #transform c2 to overlay with c1
	# c2 += true_traj[(idx+skip)*100] - true_traj[(idx)*100] #works better(?)
	# c2 += noise_scale*np.random.randn(np.shape(c2)[0],3)

	# shift_scale = 0.0 #standard deviation by which to shift the grid BEFORE SAMPLING corresponding segments of the point cloud
	# shift = tf.cast(tf.constant([shift_scale*tf.random.normal([1]).numpy()[0], shift_scale*tf.random.normal([1]).numpy()[0], 0.2*shift_scale*tf.random.normal([1]).numpy()[0], 0, 0, 0]), tf.float32)

	# x0 = tf.constant([0., 0., 0., 0., 0., 0.])
	# it = ICET(cloud1 = c1, cloud2 = c2, fid = 70, niter = 1, draw = False, group = 2, 
	# 	RM = False, DNN_filter = False, cheat = x0+shift)
	# # --------------------------------------------------------------------------

	#Get ragged tensor containing all points from each scan inside each sufficient voxel
	in1 = it.inside1
	npts1 = it.npts1
	in2 = it.inside2
	npts2 = it.npts2
	corr = it.corr #indices of bins that have enough points from scan1 and scan2

	#get indices of rag with >= n elements
	ncells = tf.shape(corr)[0].numpy() #num of voxels with sufficent number of points
	enough1 = tf.gather(in1, corr)
	enough2 = tf.gather(in2, corr)

	for j in range(numShifts):
		#init array to store indices
		idx1 = np.zeros([ncells ,npts])
		idx2 = np.zeros([ncells ,npts])

		#loop through each element of ragged tensor
		for i in range(ncells):
		    idx1[i,:] = tf.random.shuffle(enough1[i])[:npts].numpy() #shuffle order and take first N elements
		    idx2[i,:] = tf.random.shuffle(enough2[i])[:npts].numpy() #shuffle order and take first N elements

		idx1 = tf.cast(tf.convert_to_tensor(idx1), tf.int32) #indices in scan 1 of points we've selected
		idx2 = tf.cast(tf.convert_to_tensor(idx2), tf.int32) 

		from1 = tf.gather(it.cloud1_tensor, idx1)
		from2 = tf.gather(it.cloud2_tensor, idx2) 	   #transformed by "ground truth" translation

		scan1 = tf.reshape(from1, [-1, 3]).numpy()
		scan2 = tf.reshape(from2, [-1, 3]).numpy()

		#randomly translate each sample from scan 2
		rand = tf.constant([1.0, 1.0, 0.1])*tf.random.normal([ncells, 3])
		#tile and apply to scan2
		t = tf.tile(rand, [npts,1])
		t = tf.reshape(tf.transpose(t), [3,npts,-1])
		t = tf.transpose(t, [2,1,0])
		t = tf.reshape(t, [-1, 3])
		scan2 += t.numpy()

		rand += shift[:3]

		#NEW: suppress extended axis -----------------------------------
		# print("OG U", tf.shape(it.U))

		LUT = tf.matmul(it.L, tf.transpose(it.U, [0,2,1]))
		# print("-- LUT", tf.shape(LUT))

		# dz_new = tf.matmul(LUT, dnnsoln[:,:,None])
		rand_compact = tf.matmul(LUT, rand[:,:,None])
		rand_compact_xyz = tf.matmul(it.U, rand_compact)

		# print("\n LUT", tf.shape(LUT))
		#---------------------------------------------------------------

		# print("before \n", np.shape(it.corn.numpy()))
		# print("after \n", it.s2c(it.corn.numpy()))

		if idx*(j+1) == start_idx:
			scan1_cum = scan1
			scan2_cum = scan2
			rand_cum = rand #old - use for D2D vs DNN notebook
			rand_cum_compact = rand_compact_xyz #only compact
			LUT_cum = LUT.numpy()
			L_cum = it.L.numpy()
			U_cum = it.U.numpy()
			corn_cum = it.corn.numpy()
			# corn_cum = it.s2c(it.corn.numpy()) #test
		else:
			scan1_cum = np.append(scan1_cum, scan1, axis = 0)
			scan2_cum = np.append(scan2_cum, scan2, axis = 0)
			rand_cum = np.append(rand_cum, rand, axis = 0) #old - use for D2D vs DNN notebook
			rand_cum_compact = np.append(rand_cum_compact, rand_compact_xyz, axis = 0) #overwrites soln with only compact component
			LUT_cum = np.append(LUT_cum, LUT.numpy(), axis = 0)
			U_cum = np.append(U_cum, it.U.numpy(), axis = 0)
			L_cum = np.append(L_cum, it.L.numpy(), axis = 0)
			corn_cum = np.append(corn_cum, it.corn.numpy(), axis = 0)
			# corn_cum = np.append(corn_cum, it.s2c(it.corn.numpy()), axis = 0) #test

		# print("\n rand_cum", tf.shape(rand_cum))
		# print("\n rand_cum_compact", tf.shape(rand_cum_compact))



	print("got", tf.shape(enough2.to_tensor())[0].numpy()*numShifts, "training samples from scan", idx)

#small files
# np.save('perspective_shift/training_data/compact_scan1', scan1_cum)
# np.save('perspective_shift/training_data/compact_scan2', scan2_cum)
# np.save('perspective_shift/training_data/compact_ground_truth', rand_cum_compact)
# np.save('perspective_shift/training_data/ground_truth', rand_cum)
# np.save('perspective_shift/training_data/LUT', LUT_cum) 
# np.save('perspective_shift/training_data/L', L_cum)
# np.save('perspective_shift/training_data/U', U_cum)
# np.save('perspective_shift/training_data/corn', corn_cum)

#big files
np.save('/media/derm/06EF-127D1/TrainingData/compact/0117_compact_scan1', scan1_cum)
np.save('/media/derm/06EF-127D1/TrainingData/compact/0117_compact_scan2', scan2_cum)
np.save('/media/derm/06EF-127D1/TrainingData/compact/0117_compact_ground_truth', rand_cum_compact)
np.save('/media/derm/06EF-127D1/TrainingData/compact/0117_ground_truth', rand_cum)
np.save('/media/derm/06EF-127D1/TrainingData/compact/0117_LUT', LUT_cum)
np.save('/media/derm/06EF-127D1/TrainingData/compact/0117_L', L_cum)
np.save('/media/derm/06EF-127D1/TrainingData/compact/0117_U', U_cum)
np.save('/media/derm/06EF-127D1/TrainingData/compact/0117_corn', corn_cum)

#0091: 0-300, skip3
#0095: 75-175, skip3
#KCtown3: 1400-1700, 
#KCtown2: 3200-3600, skip ~ N(0,3)
#KCtown1: 2300-2700