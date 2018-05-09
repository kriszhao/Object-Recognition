import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(im, samples):
	variations = []
	variations.append(im)

	# # gaussian filter
	# for i in range(samples):
	# 	kernNum = i
	# 	variations.append(cv2.GaussianBlur(im,(kernNum,kernNum),0))

	# median filter
	for i in range(samples):
		kernNum = i
		variations.append(cv2.medianBlur(im,kernNum))

	# make hue more yellow (natural light)
	for i in range(samples):
		copy = np.copy(im)
		gain = 1./samples
		copy[:,:,0] = copy[:,:,0]*(1.-gain*i)
		variations.append(copy)

	# saturation

	# contrast

	# rotate
	for i in range(samples):
		deg = 360/samples
		rows,cols,rgb = im.shape
		M = cv2.getRotationMatrix2D((cols/2,rows/2),i*deg,1)
		dst = cv2.warpAffine(im,M,(cols,rows))
		variations.append(dst)

	return variations

def runex():
	im = cv2.imread("./cut_image/result_0.jpg")
	cv2.imshow("img",im)
	cv2.waitKey(0)
	variations = preprocess(im, 5)
	for variation in variations:
		cv2.imshow("img",variation)
		cv2.waitKey(0)

runex()