# -*- coding: utf-8 -*-

from utils import *
import numpy as np
import cv2
import matplotlib.pyplot as plt

im = cv2.imread('images/rgb.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# im = mpimg.imread('images/cup1.jpg')
# imgray = rgb2gray(im)

[m, n, _] = np.shape(im)


f = np.load('f.npy')

tau = 0.1
sigma = 0.05
mu = 1000

u = primal_dual(imgray, sigma, tau, mu, f, display=True, iters=100)
# u[u>0.5] = 1
# u[u<=0.5] = 0

# u = np.array(u, dtype=np.int32)
# plt.imshow(u, cmap='gray')
# plt.show()

# # cv2.imshow('segmentation', u)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()