import cv2
import numpy as np
from student_code import my_imfilter
import matplotlib.pyplot as plt
from utils import vis_hybrid_image, load_image, save_image

filter = np.array([ [1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])
print(np.sum(filter))
print(filter)

image = 5 * np.ones((10, 10, 2), dtype=int)

filtered_image = my_imfilter(image, filter)
print(np.squeeze(filtered_image[:,:,0]))
print(np.squeeze(filtered_image[:,:,1]))

# print(arr.shape)
#
# arr = arr[..., np.newaxis]
# print(arr)
# print(arr.shape)
# for i in range(0,3):
#     for j in range(0,4):
#         print(arr[i,j,0])
# arr_dup = np.repeat(arr, 3, axis=2)
# print(arr_dup.shape)
# print(arr_dup)
#
# image1 = load_image('../data/dog.bmp')
#
# print(image1.dtype)
# print(image1.shape)
#
# plt.figure(figsize=(3, 3)); plt.imshow((image1*255).astype(np.uint8));
# #plt.show()
#
# cutoff_frequency = 7
# filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
#                                sigma=cutoff_frequency)
# filter = np.dot(filter, filter.T)
#
#
# print(type(filter))
# print(filter.shape)
#
# my_imfilter(image1, filter)