import numpy as np
import tensorflow as tf 

#limit GPU memory ---------------------------------------------------------------------
# if you don't include this TensorFlow WILL eat up all your VRAM and make rviz run poorly
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
	try:
		memlim = 2*1024
		tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memlim)])
	except RuntimeError as e:
		# print(e)
		pass
#--------------------------------------------------------------------------------------


def linear_correction_old(cloud_xyz, m_hat, period_lidar = 1):
	"""Linear correction for motion distortion, using ugly python code
	cloud_xyz: distorted cloud in Cartesian space
	m_hat: prescribed motion profile for linear correction
	period_lidar: time it takes for LIDAR sensor to record a single sweep
	"""

	period_base = (2*np.pi)/m_hat[-1]

	#remove inf values
	cloud_xyz = cloud_xyz[cloud_xyz[:,0] < 10_000]
	#convert to spherical coordinates
	cloud_spherical = c2s(cloud_xyz).numpy()

	#Because of body frame yaw rotation, we're not always doing a full roation - we need to "uncurl" initial point cloud
	# (this is NOT baked in to motion profile)
	cloud_spherical = cloud_spherical[np.argsort(cloud_spherical[:,1])] #sort by azim angle
	#get total overlap in rotation between LIDAR and base frames (since both are rotating w.r.t. world Z)
	# point of intersection = (t_intersection) * (angular velocity base)
	#						= ((n * T_a * T_b) / (T_a + T_b)) * omega_base 
	total_rot = -2*np.pi*np.sin(m_hat[-1]/(-m_hat[-1] + (2*np.pi/period_lidar))) #actually should be this!
	# print("total_rot:", total_rot)

	#scale linearly starting at theta = 0
	cloud_spherical[:,1] = ((cloud_spherical[:,1]) % (2*np.pi))*((2*np.pi - total_rot)/(2*np.pi)) + total_rot #works

	#sort by azim angle again- some points will have moved past origin in the "uncurling" process
	cloud_spherical = cloud_spherical[np.argsort(cloud_spherical[:,1])] 

	#reorient
	cloud_spherical[:,1] = ((cloud_spherical[:,1] + np.pi) % (2*np.pi)) - np.pi
	undistorted_pc = s2c(cloud_spherical).numpy() #convert back to xyz

	return undistorted_pc

def c2s(pts):
	""" converts points from cartesian coordinates to spherical coordinates """
	r = tf.sqrt(pts[:,0]**2 + pts[:,1]**2 + pts[:,2]**2)
	phi = tf.math.acos(pts[:,2]/r)
	theta = tf.math.atan2(pts[:,1], pts[:,0])

	out = tf.transpose(tf.Variable([r, theta, phi]))
	return(out)

def s2c(pts):
	"""converts spherical -> cartesian"""

	x = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.cos(pts[:,1])
	y = pts[:,0]*tf.math.sin(pts[:,2])*tf.math.sin(pts[:,1]) 
	z = pts[:,0]*tf.math.cos(pts[:,2])

	out = tf.transpose(tf.Variable([x, y, z]))
	return(out)