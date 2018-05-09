import cv2
import numpy as np

import SaliencyRC

def process(curr, thresh):
	kernel = np.ones((5,5),np.uint8)
	opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	kernel = np.ones((15,15),np.uint8)
	result = cv2.dilate(opening,kernel,iterations = 1)

	_, contours, hierarchy = cv2.findContours(result * 1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key = cv2.contourArea)
	x, y, w, h = cv2.boundingRect(contours[-1])
	target=curr[y+2:y+h-2,x+2:x+w-2]

	return target

def saliency_ft(curr):
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

	return process(curr, thresh)

def saliency_rc(curr):
	modi = curr.copy()
	mod = modi.astype(np.float32)
	mod *= 1.0 / 255.0
	sal = SaliencyRC.GetRC(mod, 20, 200)
	idxs = np.where(sal < (sal.max()+sal.min()) / 5)
	modi[idxs] = 0
	sal = sal * 255
	sal = sal.astype(np.int16)

	imgray = cv2.cvtColor(modi,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgray,1,255,0)

	return process(curr, thresh)