from divide import divide
import matplotlib.pyplot as plt
import numpy as np
from time import sleep

for i in range(5):

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.set_xlim([-10,10])
	ax.set_ylim([-10,10])

	D = divide(fig, ax, n = 10*i+1)
	plt.show()
	sleep(1)

