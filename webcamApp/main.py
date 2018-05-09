import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detect import detect_naive
from classify import *
import numpy as np
from saliency import *

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

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
	# Scale factors
	ch, cw, _ = curr.shape
	sh, sw = (100, 100)
	scaleh, scalew = ((ch/sh), (cw/sw))

	# Saliency
	scale_curr = cv2.resize(curr, (sh, sw))
	box, cropped = saliency_ft(scale_curr)

	# Scale rectangle to original size
	box.set_height(box.get_height() * scaleh)
	box.set_width(box.get_width() * scalew)
	box.set_x(box.get_x() * scalew)
	box.set_y(box.get_y() * scaleh)
	
	# Classify
	scaled_im = scale(cropped)
	return classify_array(scaled_im), box, cropped

def main():
	cap = cv2.VideoCapture(0)
	im = plt.imshow(grab_frame(cap))
	txt = plt.text(0,0,"item name", fontsize=12)

	box = Rectangle((1,1),1,1)
	box.set_visible(False)
	currentAxis = plt.gca()
	while plt.get_fignums():
		curr = grab_frame(cap)
		im.set_data(curr)
		box.set_visible(False)

		# pure_class = just_classify(curr)
		sal_class, box, cropped = saliency_then_classify(curr)
		currentAxis.add_patch(box)
		box.set_visible(True)

		txt.set_text(sal_class)

		plt.pause(0.2)

# def runex():
# 	im = cv2.imread("./images/mouse.jpg")
# 	img = plt.imshow(im)
# 	currentAxis = plt.gca()
# 	sal_class, box, cropped = saliency_then_classify(im)
# 	currentAxis.add_patch(box)
# 	txt = plt.text(0,0,sal_class, fontsize=12)
# 	plt.show()
# 	plt.imshow(cropped)
# 	plt.show()



if __name__ == "__main__":
	# runex()
	main()

