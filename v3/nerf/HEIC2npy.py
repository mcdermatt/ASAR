from PIL import Image
from pillow_heif import register_heif_opener
from matplotlib import pyplot as plt
import numpy as np

register_heif_opener()

#loop through subdirectory
import os
rootdir = 'desk_images'

mega_image_array = np.zeros([0,1000,1000,3])

for subdir, dirs, files in os.walk(rootdir):
     for file in files:
        if file.endswith(("HEIC")):
            print(file)
            image_i = Image.open('desk_images/'+file)
            i = np.asarray(image_i)
            i = i[516:-516,12:-12,:] #crop square
            i = i[::3,::3,:]
            mega_image_array = np.append(mega_image_array, i[None,:,:,:], axis = 0)
