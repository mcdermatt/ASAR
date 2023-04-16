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

def get_H(y_j, m_hat):
	"""calculate appended Jacobian matrices H
	
		y_j: [N, 4] list of distribution cecnters in cartesian (x, y, z) 
					or homogenous coordinates (x, y, z, 1)
		m_hat: estimated acceleration [x, y, z, phi, theta, psi] 

	"""

	if np.shape(y_j)[1] == 3:
		y_j = np.append(y_j, np.ones([len(y_j),1]), axis = 1)
	# print(y_j)

	#get motion profile M using m_hat and lidar command velocity
	#get scaling time (for composite yaw rotation)
	period_lidar = 1
	t_scale = (2*np.pi)/(-m_hat[-1] + (2*np.pi/period_lidar))
	lsvec = np.linspace(0,t_scale, len(y_j))
	M = m_hat * np.array([lsvec, lsvec, lsvec, lsvec, lsvec, lsvec]).T

	#construct H matrix

	#new vectorized method ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	from numpy import sin, cos

	#decompose M into constituant vectors x, y, z, phi, theta, psi
	x = M[:,0]
	y = M[:,1]
	z = M[:,2]
	phi = M[:,3]
	theta = M[:,4]
	psi = M[:,5]

	#print output of analytic derivatives to create numpy matrices, run everything in vector form
	dT_rect_dx = np.array([[np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), -np.ones(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
	dT_rect_dx = np.transpose(dT_rect_dx, (2,0,1)) #reorder to (N, 4, 4)  
	H_x = dT_rect_dx @ y_j[:,:,None] #multiply by vector of points under consideration
	H_x = np.reshape(H_x, (-1,1)) #reshape to (4N x 1)

	dT_rect_dy = np.array([[np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), -np.ones(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
	dT_rect_dy = np.transpose(dT_rect_dy, (2,0,1)) #reorder to (N, 4, 4)  
	H_y = dT_rect_dy @ y_j[:,:,None] #multiply by vector of points under consideration
	H_y = np.reshape(H_y, (-1,1)) #reshape to (4N x 1)


	dT_rect_dz = np.array([[np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), -np.ones(len(x))],
						   [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
	dT_rect_dz = np.transpose(dT_rect_dz, (2,0,1)) #reorder to (N, 4, 4)  
	H_z = dT_rect_dz @ y_j[:,:,None] #multiply by vector of points under consideration
	H_z = np.reshape(H_z, (-1,1)) #reshape to (4N x 1)

	dT_rect_dphi = np.array([[np.zeros(len(x)), (sin(phi)*sin(psi)*sin(theta)**2 + sin(phi)*sin(psi)*cos(theta)**2 + sin(theta)*cos(phi)*cos(psi))/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), (-sin(phi)*sin(theta)*cos(psi) + sin(psi)*sin(theta)**2*cos(phi) + sin(psi)*cos(phi)*cos(theta)**2)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), (-sin(phi)*sin(theta)**2*cos(psi) - sin(phi)*cos(psi)*cos(theta)**2 + sin(psi)*sin(theta)*cos(phi))/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), (-sin(phi)*sin(psi)*sin(theta) - sin(theta)**2*cos(phi)*cos(psi) - cos(phi)*cos(psi)*cos(theta)**2)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), cos(phi)*cos(theta)/(sin(phi)**2*sin(theta)**2 + sin(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(phi)**2*cos(theta)**2), -sin(phi)*cos(theta)/(sin(phi)**2*sin(theta)**2 + sin(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(phi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
	dT_rect_dphi = np.transpose(dT_rect_dphi, (2,0,1)) #reorder to (N, 4, 4)  
	H_phi = dT_rect_dphi @ y_j[:,:,None] #multiply by vector of points under consideration
	H_phi = np.reshape(H_phi, (-1,1)) #reshape to (4N x 1)

	dT_rect_dtheta = np.array([[-sin(theta)*cos(psi)/(sin(psi)**2*sin(theta)**2 + sin(psi)**2*cos(theta)**2 + sin(theta)**2*cos(psi)**2 + cos(psi)**2*cos(theta)**2), sin(phi)*cos(psi)*cos(theta)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), cos(phi)*cos(psi)*cos(theta)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [-sin(psi)*sin(theta)/(sin(psi)**2*sin(theta)**2 + sin(psi)**2*cos(theta)**2 + sin(theta)**2*cos(psi)**2 + cos(psi)**2*cos(theta)**2), sin(phi)*sin(psi)*cos(theta)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), sin(psi)*cos(phi)*cos(theta)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [-cos(theta)/(sin(theta)**2 + cos(theta)**2), -sin(phi)*sin(theta)/(sin(phi)**2*sin(theta)**2 + sin(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(phi)**2*cos(theta)**2), -sin(theta)*cos(phi)/(sin(phi)**2*sin(theta)**2 + sin(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2 + cos(phi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
	dT_rect_dtheta = np.transpose(dT_rect_dtheta, (2,0,1)) #reorder to (N, 4, 4)  
	H_theta = dT_rect_dtheta @ y_j[:,:,None] #multiply by vector of points under consideration
	H_theta = np.reshape(H_theta, (-1,1)) #reshape to (4N x 1)

	dT_rect_dpsi = np.array([[-sin(psi)*cos(theta)/(sin(psi)**2*sin(theta)**2 + sin(psi)**2*cos(theta)**2 + sin(theta)**2*cos(psi)**2 + cos(psi)**2*cos(theta)**2), (-sin(phi)*sin(psi)*sin(theta) - sin(theta)**2*cos(phi)*cos(psi) - cos(phi)*cos(psi)*cos(theta)**2)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), (sin(phi)*sin(theta)**2*cos(psi) + sin(phi)*cos(psi)*cos(theta)**2 - sin(psi)*sin(theta)*cos(phi))/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [cos(psi)*cos(theta)/(sin(psi)**2*sin(theta)**2 + sin(psi)**2*cos(theta)**2 + sin(theta)**2*cos(psi)**2 + cos(psi)**2*cos(theta)**2), (sin(phi)*sin(theta)*cos(psi) - sin(psi)*sin(theta)**2*cos(phi) - sin(psi)*cos(phi)*cos(theta)**2)/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), (sin(phi)*sin(psi)*sin(theta)**2 + sin(phi)*sin(psi)*cos(theta)**2 + sin(theta)*cos(phi)*cos(psi))/(sin(phi)**2*sin(psi)**2*sin(theta)**2 + sin(phi)**2*sin(psi)**2*cos(theta)**2 + sin(phi)**2*sin(theta)**2*cos(psi)**2 + sin(phi)**2*cos(psi)**2*cos(theta)**2 + sin(psi)**2*sin(theta)**2*cos(phi)**2 + sin(psi)**2*cos(phi)**2*cos(theta)**2 + sin(theta)**2*cos(phi)**2*cos(psi)**2 + cos(phi)**2*cos(psi)**2*cos(theta)**2), np.zeros(len(x))], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))], [np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))]])
	dT_rect_dpsi = np.transpose(dT_rect_dpsi, (2,0,1)) #reorder to (N, 4, 4)  
	H_psi = dT_rect_dpsi @ y_j[:,:,None] #multiply by vector of points under consideration
	H_psi = np.reshape(H_psi, (-1,1)) #reshape to (4N x 1)

	H = np.array([H_x, H_y, H_z, H_phi, H_theta, H_psi]).T[0]


	# # loopy analytic method:  VERY SLOW ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# from sympy import *
	# init_printing(use_unicode=True)

	# x, y, z, phi, theta, psi = symbols('x y z phi theta psi')

	# T_roll = Matrix([[1, 0, 0, 0],
	#                  [0, cos(phi), sin(phi), 0],
	#                  [0, -sin(phi), cos(phi), 0],
	#                  [0, 0, 0, 1]])
	
	# T_pitch = Matrix([[cos(theta), 0, -sin(theta), 0],
	#                  [0, 1, 0, 0],
	#                  [sin(theta), 0, cos(theta), 0],
	#                  [0, 0, 0, 1]])
	
	# T_yaw = Matrix([[cos(psi), sin(psi), 0, 0],
	#                  [-sin(psi), cos(psi), 0, 0],
	#                  [0, 0, 1, 0],
	#                  [0, 0, 0, 1]])
	
	# T_trans = Matrix([[1, 0, 0, x],
	#                  [0, 1, 0, y],
	#                  [0, 0, 1, z],
	#                  [0, 0, 0, 1]])
	
	# # T_rect = (T_roll * T_pitch * T_yaw * T_trans)
	# T_rect = (T_roll * T_pitch * T_yaw * T_trans).inv()
	# # pprint(T_rect)

	# #get all partial deriviatives
	# dT_rect_dx = diff(T_rect, x)
	# dT_rect_dy = diff(T_rect, y)
	# dT_rect_dz = diff(T_rect, z)
	# dT_rect_dphi = diff(T_rect, phi)
	# dT_rect_dtheta = diff(T_rect, theta)
	# dT_rect_dpsi = diff(T_rect, psi)

	# H = np.zeros([0,6])
	# for p in range(len(y_j)):

	# 	# print("y_j[p]", y_j[p])
	# 	# print("M[p]", M[p])

	# 	#for each component of H_i
	# 	#1) substitute in values of motion profile M into partial derivatives
	# 	#2) convert back from SymPy analytical formulation back to np array
	# 	#3) multiply by corresponding point centers

	# 	H_x = np.array(dT_rect_dx.subs([(x, M[p,0]),
	# 									(y, M[p,1]),
	# 									(z, M[p,2]),
	# 									(phi, M[p,3]),
	# 									(theta, M[p,4]),
	# 									(psi, M[p,5])])).astype(np.float64) @ y_j[p]
	# 	H_y = np.array(dT_rect_dy.subs([(x, M[p,0]),
	# 									(y, M[p,1]),
	# 									(z, M[p,2]),
	# 									(phi, M[p,3]),
	# 									(theta, M[p,4]),
	# 									(psi, M[p,5])])).astype(np.float64) @ y_j[p]
	# 	H_z = np.array(dT_rect_dz.subs([(x, M[p,0]),
	# 									(y, M[p,1]),
	# 									(z, M[p,2]),
	# 									(phi, M[p,3]),
	# 									(theta, M[p,4]),
	# 									(psi, M[p,5])])).astype(np.float64) @ y_j[p]
	# 	H_phi = np.array(dT_rect_dphi.subs([(x, M[p,0]),
	# 									(y, M[p,1]),
	# 									(z, M[p,2]),
	# 									(phi, M[p,3]),
	# 									(theta, M[p,4]),
	# 									(psi, M[p,5])])).astype(np.float64) @ y_j[p]
	# 	H_theta = np.array(dT_rect_dtheta.subs([(x, M[p,0]),
	# 									(y, M[p,1]),
	# 									(z, M[p,2]),
	# 									(phi, M[p,3]),
	# 									(theta, M[p,4]),
	# 									(psi, M[p,5])])).astype(np.float64) @ y_j[p]
	# 	H_psi = np.array(dT_rect_dpsi.subs([(x, M[p,0]),
	# 									(y, M[p,1]),
	# 									(z, M[p,2]),
	# 									(phi, M[p,3]),
	# 									(theta, M[p,4]),
	# 									(psi, M[p,5])])).astype(np.float64) @ y_j[p]

	# 	H_i = np.array([H_x, H_y, H_z, H_phi, H_theta, H_psi]).T

	# 	H = np.append(H, H_i, axis = 0)

	# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	return H

def linear_correction_old(cloud_xyz, m_hat, period_lidar = 1):
	"""Linear correction for motion distortion, using ugly python code

	cloud_xyz: distorted cloud in Cartesian space
	m_hat: estimated motion profile for linear correction
	period_lidar: time it takes for LIDAR sensor to record a single sweep
	"""

	period_base = (2*np.pi)/m_hat[-1]

	#remove inf values
	cloud_xyz = cloud_xyz[cloud_xyz[:,0] < 10_000]
	#convert to spherical coordinates
	# cloud_spherical = c2s(cloud_xyz).numpy()

	#Because of body frame yaw rotation, we're not always doing a full roation - we need to "uncurl" initial point cloud
	# (this is NOT baked in to motion profile)
	# cloud_spherical = cloud_spherical[np.argsort(cloud_spherical[:,1])] #sort by azim angle
	#get total overlap in rotation between LIDAR and base frames (since both are rotating w.r.t. world Z)
	# point of intersection = (t_intersection) * (angular velocity base)
	#						= ((n * T_a * T_b) / (T_a + T_b)) * omega_base 
	# total_rot = -2*np.pi*np.sin(m_hat[-1]/(-m_hat[-1] + (2*np.pi/period_lidar)))
	# print("total_rot:", total_rot)

	#scale linearly starting at theta = 0
	# cloud_spherical[:,1] = ((cloud_spherical[:,1]) % (2*np.pi))*((2*np.pi - total_rot)/(2*np.pi)) + total_rot #works

	#sort by azim angle again- some points will have moved past origin in the "uncurling" process
	# cloud_spherical = cloud_spherical[np.argsort(cloud_spherical[:,1])] 

	#reorient
	# cloud_spherical[:,1] = ((cloud_spherical[:,1] + np.pi) % (2*np.pi)) - np.pi
	# cloud_xyz = s2c(cloud_spherical).numpy() #convert back to xyz

	rectified_vel  = -m_hat[None,:]
	# rectified_vel[0,-1] = 0 #zero out yaw since we already compensated for it

	T = (2*np.pi)/(-m_hat[-1] + (2*np.pi/period_lidar)) #time to complete 1 scan #was this
	# print(T)
	rectified_vel = rectified_vel * T #was this
	# print(rectified_vel[:,-1])
	# rectified_vel[:-1] = rectified_vel[:-1] * T #nope
	# rectified_vel[:,-1] = rectified_vel[:,-1] * T #also nope

	#TODO: is this is a bad way of doing it? ... what happens if most of the points are on one half of the scene??
	part2 = np.linspace(0.5, 1.0, len(cloud_xyz)//2)[:,None]
	part1 = np.linspace(0, 0.5, len(cloud_xyz) - len(cloud_xyz)//2)[:,None]
	motion_profile = np.append(part1, part2, axis = 0) @ rectified_vel  
	# print(motion_profile)

	#Apply motion profile~~~~~~~~~~~~
	T = []
	for i in range(len(motion_profile)):
		tx, ty, tz, roll, pitch, yaw = motion_profile[i]
		R = np.dot(np.dot(np.array([[1, 0, 0], 
									[0, np.cos(roll), -np.sin(roll)], 
									[0, np.sin(roll), np.cos(roll)]]), 
						np.array([[np.cos(pitch), 0, np.sin(pitch)], 
								  [0, 1, 0], 
								  [-np.sin(pitch), 0, np.cos(pitch)]])), 
						np.array([[np.cos(yaw), -np.sin(yaw), 0], 
								  [np.sin(yaw), np.cos(yaw), 0], 
								  [0, 0, 1]]))
		T.append(np.concatenate((np.concatenate((R, np.array([[tx], [ty], [tz]])), axis=1), np.array([[0, 0, 0, 1]])), axis=0))
	
	#should be the same size:
	# print(len(T))
	# Apply inverse of motion transformation to each point
	undistorted_pc = np.zeros_like(cloud_xyz)
	for i in range(len(cloud_xyz)):
		point = np.concatenate((cloud_xyz[i], np.array([1])))
		T_inv = np.linalg.inv(T[i])
		corrected_point = np.dot(T_inv, point)[:3]
		undistorted_pc[i] = corrected_point
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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