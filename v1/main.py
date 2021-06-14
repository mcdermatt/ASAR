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
#	how many seconds since enemy was last seen
#	player health


# Netork Architecture- DDPG vs RNN
#	DDPG
#		more complex network structures
#		run from scratch each time
#
#	RNN
#		single network (GRU in TF)
#		REQUIRES CORPUS (bad)

#init ------------------

	#init map

	#place player and enemy

	#init network W&B 
	#	start from checkpoint or no

#main ------------------

	#if 