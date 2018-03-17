# -*- coding: utf-8 -*-

import numpy as np
from utils import *
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('images/cup1.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im = mpimg.imread('images/cup1.jpg')
# imgray = rgb2gray(im)

[m, n, _] = np.shape(im)

white = np.zeros((m, n, 3))
white[:, :, 0] = 255
white[:, :, 1] = 255
white[:, :, 2] = 255

black = np.zeros((m, n, 3))
black[:, :, 0] = 0
black[:, :, 1] = 0
black[:, :, 2] = 0

d1 = im - white
f1 = np.sqrt(d1[:, :, 0] ** 2 + d1[:, :, 1] ** 2 + d1[:, :, 2] ** 2)
d2 = im - black
f2 = np.sqrt(d2[:, :, 0] ** 2 + d2[:, :, 1] ** 2 + d2[:, :, 2] ** 2)

f = f1 - f2

u = np.ones((m, n)) * 0.1
v = np.zeros((m, n))
c = np.ones((m, n))

f_flat = np.reshape(f, (1, m*n))
u_flat = np.reshape(f, (1, m*n))
c_uv_flat = np.reshape(c*(u-v), (1, m*n))
uv_flat = np.reshape((u-v), (1, m*n))

result = np.dot(f_flat, u_flat.T) + np.dot(uv_flat, c_uv_flat.T)
print(result)



tau = 0.1
sigma = 0.05
mu = 1000

u = primal_dual(imgray, sigma, tau, mu, f, iters=12)
u[u>0.5] = 1
u[u<=0.5] = 0

u = np.array(u, dtype=np.int32)
plt.imshow(u, cmap='gray')
plt.show()

# # cv2.imshow('segmentation', u)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()