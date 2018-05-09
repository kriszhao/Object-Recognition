import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def detect_naive(im):
	""" Returns inner image and bound box """
	roi1, slices1 = find_comp(im, bright=True)
	roi2, slices2 = find_comp(im, bright=False)

	if np.sum(roi1.shape) < np.sum(roi2.shape):
		return roi1, slices1
	return roi2, slices2

def find_comp(im, bright=True):
	if bright:
		mask = im > im.mean()
	else:
		mask = im < im.mean()

	label_im, nb_labels = ndimage.label(mask)

	# Find the largest connected component
	sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
	mask_size = sizes < 1000
	remove_pixel = mask_size[label_im]
	label_im[remove_pixel] = 0
	labels = np.unique(label_im)
	label_im = np.searchsorted(labels, label_im)

	# Now that we have only one connected component, extract it's bounding box
	slice_x, slice_y = ndimage.find_objects(label_im)[0]
	roi = im[slice_x, slice_y]

	return roi, (slice_x, slice_y)

	# plt.figure(figsize=(4, 2))
	# plt.axes([0, 0, 1, 1])
	# plt.imshow(roi)
	# plt.axis('off')

	# plt.show()	