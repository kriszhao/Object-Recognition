import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detect import detect_naive
from classify import *
import numpy as np
import SaliencyRC

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

def ft_process(curr, thresh):
	kernel = np.ones((5,5),np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	kernel = np.ones((15,15),np.uint8)
	result = cv2.dilate(opening,kernel,iterations = 1)

	_, contours, hierarchy = cv2.findContours(result * 1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key = cv2.contourArea)
	x, y, w, h = cv2.boundingRect(contours[-1])
	target=curr[y+2:y+h-2,x+2:x+w-2]

	rect = Rectangle((y+2, x+2),\
	 h, w, linewidth=5, \
	 edgecolor="red", fill=False)

	return rect, target

def saliency(curr):
	modi = curr.copy()
	mod = modi.astype(np.float32)
	mod *= 1.0 / 255.0
	sal = SaliencyRC.GetFT(mod)
	idxs = np.where(sal < (sal.max()+sal.min()) / 5)
	modi[idxs] = 0
	sal = sal * 255
	sal = sal.astype(np.int16)

	imgray = cv2.cvtColor(modi,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,1,255,0)

	return ft_process(curr, thresh)

def just_classify(file_name):
	curr = cv2.imread(file_name)

	# Classify
	scaled_im = cv2.resize(curr, (299, 299))
	scaled_im = scaled_im.astype(np.float32)
	scaled_im *= 1.0 / 255

	return classify_array(scaled_im)

def saliency_then_classify(file_name):
	curr = cv2.imread(file_name)

	# Saliency
	box, cropped = saliency(curr)
	cv2.imshow(file_name + " cropped", cropped)
	cv2.imwrite(file_name + "_cropped.jpg", cropped)
	
	# Classify
	scaled_im = cv2.resize(cropped, (299, 299))
	return classify_array(scaled_im)

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