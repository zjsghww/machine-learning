#!/bin/bash/env python
#-*- coding: utf-8
__author__ = 'ZhangJin'

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def func(x, y, sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))


suanzi1 = np.fromfunction(func, (5, 5), sigma=5)

# Laplace扩展算子
suanzi2 = np.array([[1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]])


def imconv(image_array, suanzi):
    image = image_array.copy()
    dim1, dim2 = image.shape
    for i in range(1, dim1-1):
        for j in range(1, dim2-1):
            image[i,j] = (image_array[(i-1):(i+2),(j-1):(j+2)]*suanzi).sum()
    #归一化
    image = image * (255.0/np.amax(image))
    return image

suanzi = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

image = Image.open("lena_std.tif").convert("L")
image_array = np.array(image)
image2 = imconv(image_array, suanzi)
plt.subplot(2,1,1)
plt.imshow(image_array,cmap="gray")
plt.axis("off")
plt.subplot(2,1,2)
plt.imshow(image2,cmap="gray")
plt.axis("off")
plt.show()


image_blur = signal.convolve2d(image_array, suanzi1, mode="same")
image2 = signal.convolve2d(image_blur, suanzi2, mode="same")
image2 = (image2/float(image2.max()))*255
image2[image2>image2.mean()] = 255


image_y = imconv(image_array, suanzi_y)
image_xy = np.sqrt(image_x**2+image_y**2)
image_xy = image_xy * (255.0/np.amax(image_xy))

