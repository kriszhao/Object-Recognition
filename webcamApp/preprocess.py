import cv2
import matplotlib.pyplot as plt

def preprocess(im, samples):
	variations = []
	variations.append(im)

	# gaussian filter
	for i in range(samples):
		kernNum = i
		variations.append(cv2.GaussianBlur(im,(kernNum,kernNum),0))

	# median filter
	for i in range(samples):
		kernNum = i
		variations.append(cv2.medianBlur(im,kernNum))

	# hues

	# saturation

	# contrast

	# rotate

	return variations