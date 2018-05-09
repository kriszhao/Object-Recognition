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

def just_classify(file_name):
	curr = cv2.imread(file_name)

	# Classify
	scaled_im = scale(curr)
	return classify_array(scaled_im)

def saliency_then_classify(file_name):
	curr = cv2.imread(file_name)

	# Saliency
	box, cropped = saliency_rc(curr)
	cv2.imwrite(file_name + "_cropped.jpg", cropped)
	
	# Classify
	scaled_im = scale(cropped)
	return classify_array(scaled_im)

def compare_presalienced(file_base, file_name):
	path = file_base + file_name
	if file_name.endswith("_cropped.jpg"):
		print("Classification with saliency predicts ", file_name, " is a " + just_classify(path))
	else:
		print("Pure classification predicts ", file_name, " is a " + just_classify(path))

def compare(file_base, file_name):
	path = file_base + file_name
	print("Pure classification predicts ", file_name, " is a " + just_classify(path))
	print("Classification with saliency predicts ", file_name, " is a " + saliency_then_classify(path))

def main():
	file_base = "images/"
	directory = os.fsencode(file_base)
	for file in os.listdir(directory):
		file_name = os.fsdecode(file)
		if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".gif") or file_name.endswith(".bmp"):
			compare(file_base, file_name)


if __name__ == "__main__":
	# img = cv2.imread('flwr.jpg',0)
	# draw_box(img)
	main()