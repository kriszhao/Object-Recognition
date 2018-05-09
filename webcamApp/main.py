import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detect import detect_naive
from classify import *
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

def main():
	cap = cv2.VideoCapture(0)

	im = plt.imshow(grab_frame(cap))
	# plt.text(0,0,"item name")
	# plt.ion()

	box = Rectangle((1,1),1,1)
	box.set_visible(False)

	# scaled_im = cv2.resize(grab_frame(cap), (32,32))
	# plt.imshow(scaled_im)

	while True:
		curr = grab_frame(cap)
		im.set_data(curr)
		curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
		box.set_visible(False)
		box, slices = draw_box(curr_gray)
		slice_x, slice_y = slices
		box.set_visible(True)


		scaled_im = cv2.resize(curr[slice_x, slice_y], (299,299))
		# variations = preprocess(scaled_im)
		#print(classify_array(scaled_im))

		plt.pause(0.2)

	# plt.ioff()
	# plt.show()


if __name__ == "__main__":
	# img = cv2.imread('flwr.jpg',0)
	# draw_box(img)
	main()