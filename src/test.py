import cv2
import numpy as np
import numpy.fft as fft
from student_code import my_imfilter, create_hybrid_image
import matplotlib.pyplot as plt
from utils import vis_hybrid_image, load_image, save_image


def plot_image(image):
	plt.figure(figsize=(4, 4))
	plt.imshow((image * 255).astype(np.uint8))

def test_my_filter(filter):
	image1 = load_image('../data/dog.bmp')
	filtered = my_imfilter(image1, filter)
	plot_image(filtered)

def test_hybrid_image(filter):
	image1 = load_image('../data/dog.bmp')
	image2 = load_image('../data/cat.bmp')
	low, high, hybrid = vis_hybrid_image(image1, image2, filter)
	plot_image(high)

def test_fft_blur(image, filter):
	im_row, im_col, im_height = np.shape(image)
	# print("Image dimensions: {}, {}, {}".format(im_row, im_col, im_height))
	fil_row, fil_col = np.shape(filter)
	# print("Filter dimensions: {}, {}".format(fil_row, fil_col))

	# Padding the filter to match image dimensions
	if (im_row - fil_row) % 2 == 0:
		row_pad = ((im_row - fil_row) // 2, (im_row - fil_row) // 2)
	else:
		row_pad = ((im_row - fil_row) // 2 + 1, (im_row - fil_row) // 2)
	if (im_col - fil_col) % 2 == 0:
		col_pad = ((im_col - fil_col) // 2, (im_col - fil_col) // 2)
	else:
		col_pad = ((im_col - fil_col) // 2 + 1, (im_col - fil_col) // 2)
	p_filter = np.pad(filter, [row_pad, col_pad], mode="constant")
	p_fil_row, p_fil_col = np.shape(p_filter)
	# print("Padded filter dimensions: {}, {}".format(p_fil_row, p_fil_col))

	# Blurring
	fft_filter = fft.fft2(p_filter)
	fft_image = np.ndarray((im_row, im_col, im_height), dtype="complex64")  # fft output is complex
	mult_fft = np.ndarray((im_row, im_col, im_height), dtype="complex128")  # multiplying 2 64bit numbers can be 128bits
	conv = np.ndarray((im_row, im_col, im_height), dtype="complex128")
	for i in range(0, im_height):
		fft_image[:,:,i] = fft.fft2(image[:,:,i])
		mult_fft[:,:,i] = np.multiply(fft_image[:,:,i], fft_filter)
		conv[:,:,i] = fft.ifftshift(fft.ifft2(mult_fft[:,:,i]))

	plot_image(conv)
	return conv

def test_fft_sharpen(image, low):
	assert np.shape(image) == np.shape(low)
	high = image - low + 0.5
	plot_image(high)
	return high

def main():
	cutoff_frequency = 7
	filter = cv2.getGaussianKernel(ksize=cutoff_frequency * 4 + 1,
	                               sigma=cutoff_frequency)
	filter = np.dot(filter, filter.T)

	image1 = load_image('../data/dog.bmp')
	image2 = load_image('../data/cat.bmp')
	plot_image(image1)

	# Testing my_filter function
	test_my_filter(filter)

	# Testing vis_hybrid_image function
	test_hybrid_image(filter)

	# Testing the blurring and sharpening through fft
	low = test_fft_blur(image1, filter)
	high = test_fft_sharpen(image1, low)
	hybrid = high - 0.5 + low
	plot_image(hybrid)

	# Testing create_hybrid_image function
	low, high, hybrid = create_hybrid_image(image1, image2, filter)
	plot_image(low)
	plot_image(high)
	plot_image(hybrid)
	plt.show()


main()