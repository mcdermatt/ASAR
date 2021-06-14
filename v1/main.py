import cv2
import numpy as np
from matplotlib import pyplot as plt
from player import Player
import time

#input data from enviornment:
#	lidar points (x100)
#		in later version, include enemy body in lidar (currently just has map)
#	heading
#	global position
#	seen enemy position(s)
#	how many seconds since enemy was last seen??
#	player health
#	NEED TO REPRESENT VELOCITY
#		save past n frames of Lidar??


# Netork Architecture- DDPG vs RNN
#	DDPG
#		more complex network structures
#		run from scratch each time
#
#	RNN
#		single network (LSTM/GRU in TF)
#		REQUIRES CORPUS (bad)


# Reward shaping
#	what do we want here?

#init ------------------

	#init map

	#place player and enemy

	#init network W&B 
	#	start from checkpoint or no

	# get initial state measurements from enviornment


#main ------------------

	# feed system states to actor network


	# add noise to output of actor (to help with exploration)

	# save action + noise as temp variable

	# step enviornment forward


	# get resulting state from moving agent according to network policy + noise

	# get reward from enviornment


	# send SARS' to learning function
		#use 


	# if end condition, restart simulation