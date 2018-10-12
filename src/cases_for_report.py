import cv2
import numpy as np
from student_code import my_imfilter, create_hybrid_image
from unit_tests import test_fft_blur
import matplotlib.pyplot as plt
from utils import vis_hybrid_image, load_image, save_image

def main():
	cutoff_frequency = 7
	filter = cv2.getGaussianKernel(ksize=cutoff_frequency * 4 + 1,
	                               sigma=cutoff_frequency)
	filter = np.dot(filter, filter.T)

	# Filtering example
	# dog = load_image('../data/dog.bmp')
	# dog_blurred = my_imfilter(dog, filter)
	# save_image('../results/dog_blurred.jpg', dog_blurred)

	# Black and white hybrid
	einstein = load_image('../data/einstein.bmp')
	marilyn = load_image('../data/marilyn.bmp')
	# low, high, hybrid = create_hybrid_image(marilyn, einstein, filter)
	# high = np.clip(high, 0, 255)
	# vis = vis_hybrid_image(hybrid)
	# plt.figure()
	# plt.imshow((low * 255).astype(np.uint8))
	# plt.figure()
	# plt.imshow(((high + 0.5) * 255).astype(np.uint8))
	# plt.figure(figsize=(20, 20))
	# plt.imshow(vis)
	# save_image('../results/einstein_high.jpg', high)
	# save_image('../results/einstein_high_boosted.jpg', high+0.5)
	# save_image('../results/marilyn_low.jpg', low)
	# save_image('../results/einrilyn.jpg', hybrid)
	# save_image('../results/einrilyn_vis.jpg', vis)

	# Bike Motorcycle hybrid
	bike = load_image('../data/bicycle.bmp')
	motorcycle = load_image('../data/motorcycle.bmp')
	# low, high, hybrid = create_hybrid_image(bike, motorcycle, filter)
	# high = np.clip(high, 0, 255)
	# vis = vis_hybrid_image(hybrid)
	# plt.figure()
	# plt.imshow((low * 255).astype(np.uint8))
	# plt.figure()
	# plt.imshow(((high + 0.5) * 255).astype(np.uint8))
	# plt.figure(figsize=(20, 20))
	# plt.imshow(vis)
	# save_image('../results/motorcycle_high.jpg', high)
	# save_image('../results/bike_low.jpg', low)
	# save_image('../results/motorbike.jpg', hybrid)
	# save_image('../results/motorbike_vis.jpg', vis)

	# Chicken/dinosaur hybrid
	dino = load_image('../data/dinosaur.jpg')
	chicken = load_image('../data/chicken.jpg')
	low, high, hybrid = create_hybrid_image(chicken, dino, filter)
	vis = vis_hybrid_image(hybrid)
	# plt.figure()
	# plt.imshow((low * 255).astype(np.uint8))
	# plt.figure()
	# plt.imshow(((high + 0.5) * 255).astype(np.uint8))
	# plt.figure(figsize=(20, 20))
	# plt.imshow(vis)
	save_image('../results/dino_high.jpg', high)
	save_image('../results/chicken_low.jpg', low)
	save_image('../results/chickno.jpg', hybrid)
	save_image('../results/chickno_vis.jpg', vis)

	# Convolution steps
	# sub = load_image("../data/submarine.bmp")
	# shifted, low = test_fft_blur(sub, filter)
	# save_image('../results/sub_blurred.jpg', low)
	# save_image('../results/sub_shifted.jpg', shifted)
	plt.show()

main()
