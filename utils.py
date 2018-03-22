# -*- coding: utf-8 -*-
import numpy as np


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.29899999999 * r + 0.58699999999 * g + 0.11399999999 * b
    gray = np.array(gray, dtype=np.int32)
    return gray

def dxm(u):
    M, _ = np.shape(u)
    dx = np.concatenate([u[:, :-1], np.zeros((M, 1))], axis=1) -\
            np.concatenate([np.zeros((M, 1)), u[:, :-1]], axis=1)
    return dx

def dxp(u):
    dx = np.concatenate([u[:, 1:], u[:, -1].reshape(-1, 1)], axis=1) - u
    return dx

def dym(u):
    _, N = np.shape(u)
    dy = np.concatenate([u[:-1, :], np.zeros((1, N))], axis=0) -\
            np.concatenate([np.zeros((1, N)), u[:-1, :]], axis=0)
    return dy

def dyp(u):
    dy = np.concatenate([u[1:, :], u[-1, :].reshape(1, -1)], axis=0) - u
    return dy

def primal_dual(img, sigma, tau, mu, f, iters=10):
    M, N = np.shape(img)

    p = np.ones((M, N))
    u = np.ones((M, N)) * 0.5
    ubar = np.ones((M, N)) * 0.5

    for i in range(iters):
        print(i)
        ubar_old = -u
        p1 = p - sigma * dxp(ubar)
        p2 = p - sigma * dyp(ubar)
        norm = np.maximum(np.ones((M, N)), np.sqrt(p1 ** 2 + p2 ** 2))
        p1 = p1 / norm
        p2 = p2 / norm
        p = p / norm
        u = u - tau * (dxm(p1) + dym(p2) + (1 / mu) * f)
        u = np.minimum(np.ones((M, N)), np.maximum(np.zeros((M, N)), u))
        ubar = ubar_old + 2 * u

    return u

