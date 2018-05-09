import segmentImage
import SaliencyRC
import cv2
import numpy as np
import random
import os

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images
def test_segmentation(img,n):
    img3 = img
    img3i = img3.copy()
    img3f = img3i.astype(np.float32)
    img3f *= 1. / 255
    imgLab3f = cv2.cvtColor(img3f,cv2.COLOR_BGR2LAB)
    num,imgInd = segmentImage.SegmentImage(imgLab3f,None,0.5,200,8000)

    print("a = \n",num)
    print("b = \n",imgInd)
    colors = [[random.randint(0,255),random.randint(0,255),random.randint(0,255)] for _ in range(num)]
    print("=========================================== \n")
    showImg = np.zeros(img3f.shape,dtype=np.int8)
    height = imgInd.shape[0]
    width = imgInd.shape[1]
    for y in range(height):
        for x in range(width):
            if imgInd[y,x].all() > 0:
                showImg[y,x] = colors[imgInd[y,x] % num]
    cv2.imshow("sb",showImg)
    nrootdir=("./cut_FT/")
    if not os.path.isdir(nrootdir):
      os.makedirs(nrootdir)
    cv2.imwrite( nrootdir+"seg_"+str(n)+".jpg",showImg)
    cv2.waitKey(350)
def test_rc_map(img,n):
    # img3 = cv2.imread("test1.jpg")
    img3 = img
    img3i = img3.copy()
    img3f = img3i.astype(np.float32)
    img3f *= 1. / 255
    #sal = SaliencyRC.GetRC(img3f,segK=20,segMinSize=200)
    start = cv2.getTickCount()
    sal = SaliencyRC.GetHC(img3f)
    end = cv2.getTickCount()
    print((end - start)/cv2.getTickFrequency())
    np.save("sal.npy",sal)
    # For HC 
    # idxs = np.where(sal < (sal.max()+sal.min()) / 1.5)
    idxs = np.where(sal < (sal.max()+sal.min()) / 5)
    img3i[idxs] = 0
    sal = sal * 255
    sal = sal.astype(np.int16)
    #cv2.namedWindow("sb")
    #cv2.moveWindow("sb",20,20)
    #cv2.imshow('sb',sal.astype(np.int8))
    imgray = cv2.cvtColor(img3i,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,1,255,0)
    # For HC
    # kernel = np.ones((5,5),np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # kernel = np.ones((15,15),np.uint8)
    # result = cv2.dilate(opening,kernel,iterations = 1)

    # For FT
    kernel = np.ones((5,5),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((15,15),np.uint8)
    result = cv2.dilate(opening,kernel,iterations = 1)

    contours, hierarchy = cv2.findContours(result * 1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contours[-1])
    target=img3[y+2:y+h-2,x+2:x+w-2]
    nrootdir=("./cut_FT/")
    if not os.path.isdir(nrootdir):
      os.makedirs(nrootdir)
    cv2.imwrite( nrootdir+"result_"+str(n)+".jpg",target)
    cv2.namedWindow("target")
    cv2.moveWindow("target",100,20)
    cv2.imshow("target",target)
    cv2.drawContours(img3, contours[-1], -1, (0,50,255), 2)
    cv2.rectangle(img3, (x,y), (x+w,y+h), (205,133,0), 3) 
    cv2.namedWindow("origian")
    cv2.moveWindow("origian",800,20)
    cv2.imshow("origian",img3)
    cv2.waitKey(350)

if __name__ == "__main__":
    np.set_printoptions(threshold=np.nan)
    images = load_images_from_folder("./origin")
    # person = input('Enter your name: ')
    # print('Hello', person)
    for item in range(len(images)):
    	# test_segmentation(images[item],item)
    	 test_rc_map(images[item],item)
