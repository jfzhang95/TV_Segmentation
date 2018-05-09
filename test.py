import numpy as np
import cv2
from utils import *

f = np.load('f.npy')
img = cv2.imread('images/rgb.jpg')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
u = np.ones((np.shape(imgray))) * 0
v = np.ones((np.shape(imgray))) * 0

f_flat = f.reshape(-1, 1)
u_flat = u.reshape(-1, 1)
v_flat = v.reshape(-1, 1)

MAX_ITERS = 20
theta = 1e-1
learning_rate = 1e-2
threhold = 0.5



for i in range(MAX_ITERS):
    print(i)
    # Optimization 1
    for j in range(20):
        gradient = f_flat + 2 * theta * (u_flat - v_flat)
        u_flat -= learning_rate * gradient
        u_flat = np.clip(u_flat, 0, 1)
    u = u_flat.copy().reshape((np.shape(imgray)))


    # Optimization 2
    v = solve_ROF(u, theta, iter_n=20)
    v_flat = v.copy().reshape(-1, 1)
    v[v > threhold] = 1
    v[v <= threhold] = 0
    u[u > threhold] = 1
    u[u <= threhold] = 0

    # f_flat[v.reshape(-1, 1) == 1] = -np.abs(f_flat[v.reshape(-1, 1) == 1] - 255)
    ### mode 1 ###
    # f_flat[v.reshape(-1, 1) == 0] = np.abs(f_flat[v.reshape(-1, 1) == 0] + 255)

    theta *= 10

    plt.ion()
    plt.subplot(221)
    plt.imshow(u.astype(int), cmap='gray')
    plt.title('u')
    plt.subplot(222)
    plt.imshow(v.astype(int), cmap='gray')
    plt.title('v')
    plt.subplot(223)
    plt.imshow(img)
    plt.axis('off')
    if i == MAX_ITERS-1:
        plt.show()
        plt.pause(10)
    else:
        plt.show()
        plt.pause(1)
        plt.clf()