from divide import divide
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

fig = plt.figure()
ax = fig.add_subplot()

for i in range(50):

	D = divide(fig, ax, n = i+1)

	D.ax.set_xlim([-10,10])
	D.ax.set_ylim([-10,10])

	plt.draw()
	plt.pause(0.01)
	D.ax.cla()
	D.patches = []
	D.ax.patches = []
