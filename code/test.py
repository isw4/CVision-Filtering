import cv2
import numpy as np
from student_code import my_imfilter
import matplotlib.pyplot as plt
from utils import vis_hybrid_image, load_image, save_image

arr = np.array([[1, 2, 3, 7],
                [3, 4, 3, 6],
                [4, 5, 1, 5]])

print(arr)
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

image1 = load_image('../data/dog.bmp')

print(type(image1))
print(image1.shape)

plt.figure(figsize=(3, 3)); plt.imshow((image1*255).astype(np.uint8));
#plt.show()

cutoff_frequency = 7
filter = cv2.getGaussianKernel(ksize=cutoff_frequency*4+1,
                               sigma=cutoff_frequency)
filter = np.dot(filter, filter.T)


print(type(filter))
print(filter.shape)

my_imfilter(image1, filter)