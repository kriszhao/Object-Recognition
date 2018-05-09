import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detect import detect_naive
from preprocess import preprocess

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

def draw_box(img):
	obj, coor = detect_naive(img)
	y, x = coor
	currentAxis = plt.gca()
	rect = Rectangle((x.start, y.start),\
	 x.stop - x.start, y.stop - y.start, linewidth=5, \
	 edgecolor="red", fill=False)
	currentAxis.add_patch(rect)
	return rect, obj

def main():
	cap = cv2.VideoCapture(0)

	im = plt.imshow(grab_frame(cap))

	plt.ion()

	box = Rectangle((1,1),1,1)
	box.set_visible(False)

	while True:
		curr = grab_frame(cap)
		im.set_data(curr)
		curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
		box.set_visible(False)
		box, obj = draw_box(curr_gray)
		box.set_visible(True)

		scaled_im = cv2.resize(obj, (32,32))
		variations = preprocess(scaled_im)

		plt.pause(0.2)

	# plt.ioff()
	# plt.show()


if __name__ == "__main__":
	# img = cv2.imread('flwr.jpg',0)
	# draw_box(img)
	main()