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

	currentAxis = plt.gca()
	rect = Rectangle((y+2, x+2),\
	 h, w, linewidth=5, \
	 edgecolor="red", fill=False)
	currentAxis.add_patch(rect)

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
	box, cropped = saliency(curr)
	
	# Classify
	scaled_im = scale(cropped)
	return classify_array(scaled_im), box

def main():
	cap = cv2.VideoCapture(0)
	im = plt.imshow(grab_frame(cap))
	txt = plt.text(0,0,"item name", fontsize=12)

	box = Rectangle((1,1),1,1)
	box.set_visible(False)

	# scaled_im = cv2.resize(grab_frame(cap), (32,32))
	# plt.imshow(scaled_im)

	while plt.get_fignums():
		curr = grab_frame(cap)
		im.set_data(curr)
		box.set_visible(False)
		pure_class = just_classify(curr)
		sal_class, box = saliency_then_classify(curr)
		box.set_visible(True)
		print("Pure classification predicts this object is a " + pure_class)
		print("Classification with saliency predicts this object is a " + sal_class)
		print("\n")
		txt.set_text(sal_class)


		# scaled_im = cv2.resize(curr[slice_x, slice_y], (299,299))
		# variations = preprocess(scaled_im)
		#print(classify_array(scaled_im))

		plt.pause(0.2)

	# plt.ioff()
	# plt.show()

if __name__ == "__main__":
	main()

