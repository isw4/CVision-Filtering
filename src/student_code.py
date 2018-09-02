import numpy as np
import numpy.fft as fft

def my_imfilter(image, filter):
	"""
	Apply a filter to an image. Return the filtered image.

	Args
	- image: numpy nd-array of dim (m, n, c)
	- filter: numpy nd-array of dim (k, k)
	Returns
	- filtered_image: numpy nd-array of dim (m, n, c)

	HINTS:
	- You may not use any libraries that do the work for you. Using numpy to work
	with matrices is fine and encouraged. Using opencv or similar to do the
	filtering for you is not allowed.
	- I encourage you to try implementing this naively first, just be aware that
	it may take an absurdly long time to run. You will need to get a function
	that takes a reasonable amount of time to run so that the TAs can verify
	your code works.
	- Remember these are RGB images, accounting for the final image dimension.
	"""

	(fil_row, fil_col) = filter.shape
	assert fil_row % 2 == 1
	assert fil_col % 2 == 1

	############################
	###  STUDENT CODE HERE   ###

	# Padding the image with zeros according to the size of the filter
	(im_row, im_col, im_height) = image.shape
	row_pad_size = (fil_row - 1) // 2
	col_pad_size = (fil_col - 1) // 2
	padded_image = np.pad(image, [(row_pad_size, row_pad_size), (col_pad_size, col_pad_size), (0, 0)], mode="reflect")
	# For debugging
	# print("Filter shape: {} rows, {} cols".format(fil_row, fil_col))
	# print("Image shape: {} rows, {} cols, {} height".format(im_row, im_col, im_height))
	# print("Padded image: {} rows, {} cols, {} height".format(padded_image.shape[0], padded_image.shape[1], padded_image.shape[2]))

	# Doing the filtering
	padded_filter = filter[..., np.newaxis] # increases the number of dimensions before can broadcast
	padded_filter = np.repeat(padded_filter, im_height, axis=2) # increase dimensions of the filter to match image

	filtered_image = np.zeros(image.shape, dtype=image.dtype)
	for i in range(0, im_row):
		for j in range(0, im_col):
			temp = np.multiply(padded_filter, padded_image[i:i+fil_row, j:j+fil_col, :])
			filtered_image[i, j, :] = np.sum(temp, axis=(0, 1))

	### END OF STUDENT CODE ####
	############################

	return filtered_image

def __fft_conv__(image, filter):
	"""
	Takes an image and a filter in the spatial domain and convolves them using fft

	:param image:  numpy nd-array of dim (m, n, c)
	:param filter: numpy nd-array of dim (m, n)
	:return: numpy nd-array of dim (m, n, c)
	"""
	im_row, im_col, im_height = np.shape(image)
	fil_row, fil_col = np.shape(filter)
	assert im_row == fil_row
	assert im_col == fil_col

	fft_filter = fft.fft2(filter) # FFT the filter (unshifted)
	fft_image = np.ndarray((im_row, im_col, im_height), dtype="complex64")  # fft output is complex
	mult_fft = np.ndarray((im_row, im_col, im_height), dtype="complex128")  # multiplying 2 64bit numbers can be 128bits
	conv = np.ndarray((im_row, im_col, im_height), dtype="complex128")
	for i in range(0, im_height):
		fft_image[:, :, i] = fft.fft2(image[:, :, i]) # FFT the image (unshifted)
		mult_fft[:, :, i] = np.multiply(fft_image[:, :, i], fft_filter) # Multiply by elements
		conv[:, :, i] = fft.ifftshift(fft.ifft2(mult_fft[:, :, i])) # IFFT the product, and then ifftshift
	return np.real(conv)

def create_hybrid_image(image1, image2, filter):
	"""
	Takes two images and creates a hybrid image. Returns the low
	frequency content of image1, the high frequency content of
	image 2, and the hybrid image.

	Args
	- image1: numpy nd-array of dim (m, n, c)
	- image2: numpy nd-array of dim (m, n, c)
	Returns
	- low_frequencies: numpy nd-array of dim (m, n, c)
	- high_frequencies: numpy nd-array of dim (m, n, c)
	- hybrid_image: numpy nd-array of dim (m, n, c)

	HINTS:
	- You will use your my_imfilter function in this function.
	- You can get just the high frequency content of an image by removing its low
	frequency content. Think about how to do this in mathematical terms.
	- Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
	as 'clipping'.
	- If you want to use images with different dimensions, you should resize them
	in the notebook code.
	"""

	assert image1.shape[0] == image2.shape[0]
	assert image1.shape[1] == image2.shape[1]
	assert image1.shape[2] == image2.shape[2]

	############################
	###  STUDENT CODE HERE   ###

	im_row, im_col, im_height = np.shape(image1)
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

	# Blurring Image 1
	low_frequencies = __fft_conv__(image1, p_filter)
	# Blurring Image 2
	image2_low = __fft_conv__(image2, p_filter)

	# Sharpening Image 2
	high_frequencies = image2 - image2_low

	# Hybrid Image
	hybrid_image = high_frequencies + low_frequencies
	min_hybrid = np.min(hybrid_image)
	if min_hybrid < 0: # Bring up the overall brightness to prevent underflow
		hybrid_image = hybrid_image - min_hybrid
	hybrid_image = hybrid_image / np.max(hybrid_image) # Scaling down to prevent overflow

	### END OF STUDENT CODE ####
	############################

	return low_frequencies, high_frequencies, hybrid_image
