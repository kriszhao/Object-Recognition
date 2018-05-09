import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detect import detect_naive
from classify import *
import numpy as np
from saliency import *

#from preprocess import preprocess

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

def draw_box(img):
	gray, coor = detect_naive(img)
	y, x = coor
	currentAxis = plt.gca()
	rect = Rectangle((x.start, y.start),\
	 x.stop - x.start, y.stop - y.start, linewidth=5, \
	 edgecolor="red", fill=False)
	currentAxis.add_patch(rect)
	return rect, coor

def scale(curr):
	resized = cv2.resize(curr, (299,299))
	resized = resized.astype(np.float32)
	resized *= 1.0 / 255
	return resized

def just_classify(curr):
	# Classify
	scaled_im = scale(curr)
	return classify_array(scaled_im)

def saliency_then_classify(curr):
	# Saliency
	box, cropped = saliency_ft(curr)
	
	# Classify
	scaled_im = scale(cropped)
	return classify_array(scaled_im)

def main():
	cap = cv2.VideoCapture(0)
	im = plt.imshow(grab_frame(cap))

	box = Rectangle((1,1),1,1)
	box.set_visible(False)

	while True:
		curr = grab_frame(cap)
		im.set_data(curr)
		box.set_visible(False)
		print("Pure classification predicts this object is a " + just_classify(curr))
		print("Classification with saliency predicts this object is a " + saliency_then_classify(curr))
		print("\n")
		box.set_visible(True)

		plt.pause(0.2)


if __name__ == "__main__":
	# img = cv2.imread('flwr.jpg',0)
	# draw_box(img)
	main()