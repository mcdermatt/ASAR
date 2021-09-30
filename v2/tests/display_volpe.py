from vedo import *
import numpy as np
import os

#NOTES: 
#	PCAP is a network capture(?) format from Wireshark/ network capture tool
		# https://binarymillennium.blogspot.com/2008/07/velodyne-lidar-sample-data-getting-pcap.html


location = 'C:/Users/Derm/2021-03-10-16-43-50_Velodyne-VLP-16-Data_garminSignage.txt'
cloud = np.loadtxt(open(location, "rb"), delimiter=",")
print(np.shape(cloud))
#remove all rows with NaN elements
print(np.shape(cloud))
cloud = cloud[~np.isnan(cloud).any(axis=1)]
print(np.shape(cloud))


plt1 = Plotter(N=1, axes=1, bg = (0.1,0.1,0.1), bg2 = (0.3,0.3,0.3),  interactive=True)

c = Points(cloud, c = (1,1,1), alpha = 0.5)
plt1.show(c, "Volpe Data Test", at =0).close()