# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

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

def primal_dual(img, sigma, tau, mu, f, display=False, iters=10):
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
        if display:
            u_ = u.copy()
            u_[u_ > 0.5] = 1
            u_[u_ <= 0.5] = 0
            plt.ion()
            plt.imshow(u_.astype(int), cmap='gray')
            plt.title('u')
            plt.show()
            plt.pause(0.1)
            plt.clf()

    return u



##########################################
########### For ROF Problem ##############
##########################################

def nabla(I):
    h, w = I.shape
    G = np.zeros((h, w, 2), I.dtype)
    G[:, :-1, 0] -= I[:, :-1]
    G[:, :-1, 0] += I[:, 1:]
    G[:-1, :, 1] -= I[:-1]
    G[:-1, :, 1] += I[1:]
    return G

def nablaT(G):
    h, w = G.shape[:2]
    I = np.zeros((h, w), G.dtype)
    # note that we just reversed left and right sides
    # of each line to obtain the transposed operator
    I[:, :-1] -= G[:, :-1, 0]
    I[:, 1: ] += G[:, :-1, 0]
    I[:-1]    -= G[:-1, :, 1]
    I[1: ]    += G[:-1, :, 1]
    return I

def anorm(x):
    '''Calculate L2 norm over the last array dimention'''
    return np.sqrt((x*x).sum(-1))

def project_nd(P, r):
    '''perform a pixel-wise projection onto R-radius balls'''
    nP = np.maximum(1.0, anorm(P) / r)
    return P / nP[..., np.newaxis]

def shrink_1d(X, F, step):
    '''pixel-wise scalar srinking'''
    return X + np.clip(F - X, -step, step)

def calc_energy_ROF(X, observation, clambda):
    Ereg = anorm(nabla(X)).sum()
    Edata = 0.5 * clambda * ((X - observation)**2).sum()
    return Ereg + Edata

def solve_ROF(img, clambda=4.0, iter_n=101):
    # setting step sizes and other params
    L2 = 8.0
    tau = 0.02
    sigma = 1.0 / (L2*tau)
    theta = 1.0

    X = img.copy()
    P = nabla(X)
    for i in range(iter_n):
        P = project_nd( P + sigma*nabla(X), 1.0 )
        lt = clambda * tau
        X1 = (X - tau * nablaT(P) + lt * img) / (1.0 + lt)
        X = X1 + theta * (X1 - X)
        # if i % 10 == 0:
        #     print("%.2f" % calc_energy_ROF(X, img, clambda),)
    return X

