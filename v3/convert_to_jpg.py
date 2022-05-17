from PIL import Image
import os

# File for converting provided RGB images from GNP image format (GIMP 2.0) to jpg

path ='E:/Ford/IJRR-Dataset-1-subset/IMAGES/FULL'

for x in os.listdir(path):
	if x.endswith('ppm'):
		print(x)

		fn = x
		# print(fn)

		im = Image.open(fn)
		im = im.transpose(Image.TRANSPOSE)

		newfn = x[:-4] + ".jpg"
		# print(newfn)
		im.save(newfn)