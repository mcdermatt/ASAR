import numpy as np

def p_loop(dt = 0.1, niter = 10):

	#noise covariance matrix (from NDT/ ICET) ----------------------------
	# #	taken from example output of ICET
	# Q = np.array([[3e-2, 1e-2, 3e-3],
	# 			  [1e-2, 5e-2, 3e-3],
	# 			  [3e-3, 3e-3, 1e-4]])
	# Q = Q*10 #debug

	# calculated from Volpe dataset
	# Q = np.array([[0.0987, 0.002, -0.0002],
	# 			  [0.002,  0.0358, 0.0003],
	# 			  [-0.0002, 0.0003,0.0007]])

	Q = np.array([[0.0987**2, 0.00, -0.000],
				  [0.00,  0.0358**2, 0.000],
				  [-0.000, 0.000,0.0007**2]])

	print("Q = \n", Q)
	#---------------------------------------------------------------------

	#measurement covariance matrix (from GPS/ INS)------------------------
	#this worked with ICET values
	# R = np.array([[3e-2, 1e-2, 3e-3],
	# 			  [1e-2, 3e-2, 3e-3],
	# 			  [3e-3, 3e-3, 1e-4]])
	# # R = R*100

	# R = np.identity(3)*10 #works with volpe Q for debug


	# R = np.array([[0.1829, 0, 0],
	# 			  [0, 0.6226, 0],
	# 			  [0, 0, 0.0587]])

	R = np.array([[0.05**2, 0, 0],
				  [0, 0.05**2, 0],
				  [0, 0, 0.00001**2]])


	print("R = \n", R)

	# R = np.identity(6)
	#---------------------------------------------------------------------

	# state transition model ---------------------------------------------
	#	predicts how states will evolve over time given system dynamics

	# F = np.array([[1, 0, 0, dt, 0, 0],
	# 			  [0, 1, 0, 0, dt, 0],
	# 			  [0, 0, 1, 0, 0, dt],
	# 			  [0, 0, 0, 1, 0, 0],
	# 			  [0, 0, 0, 0, 1, 0],
	# 			  [0, 0, 0, 0, 0, 1]])

	F = np.array([[1, 0, 0],
				  [0, 1, 0],
				  [0, 0, 1]])
	#---------------------------------------------------------------------

	# observation model --------------------------------------------------
	#	relates sensor readings in measurement to real world units

	H = np.array([[1, 0, 0],
				  [0, 1, 0],
				  [0, 0, 1]])

	# H = np.identity(6)

	#---------------------------------------------------------------------

	I = np.identity(3)
	# I = np.identity(6)


	sigma_x_history = np.zeros(niter)
	sigma_y_history = np.zeros(niter)
	sigma_theta_history = np.zeros(niter)

	#init P_plus
	P_plus = R
	# P_plus = np.identity(3)

	i = 0 
	while i < niter:

		#prediction step -------------------------------------------------
		
		# x_minus = F.dot(x_plus) + Gu # <- Gu is transformation estimate from NDT/ ICET 

		P_minus = F.dot(P_plus).dot(F.T) + Q
		#-----------------------------------------------------------------


		# if i > niter/2:
		# 	R = R *100

		#correction step -------------------------------------------------
		L = P_minus.dot(H.T).dot(np.linalg.pinv(H.dot(P_minus).dot(H.T) + R))

		# x_plus = x_minus + L.dot(y - H.dot(x_minus)) # y is absolute state estimates from GPS/INS

		P_plus = (I - L.dot(H)).dot(P_minus).dot((I - L.dot(H)).T) + L.dot(R).dot(L.T)
		#-----------------------------------------------------------------

		sigma_x_history[i] = np.sqrt(P_plus[0,0])
		sigma_y_history[i] = np.sqrt(P_plus[1,1])
		sigma_theta_history[i] = np.sqrt(P_plus[2,2])


		i += 1

	return(sigma_x_history, sigma_y_history, sigma_theta_history)