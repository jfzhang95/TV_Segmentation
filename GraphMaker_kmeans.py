# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import *
from sklearn.cluster import KMeans


class GraphMaker:

    foreground = 1
    background = 0

    seeds = 0
    segmented = 1

    def __init__(self):
        self.image = None
        self.graph = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.mask = None
        self.load_image('cup1.jpg')
        self.background_seeds = []
        self.foreground_seeds = []
        self.background_average = np.array(3)
        self.foreground_average = np.array(3)
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds

    def load_image(self, filename):
        self.image = cv2.imread(filename)
        self.height, self.width, _ = np.shape(self.image)
        print(self.height, self.width)
        self.graph = np.zeros_like(self.image)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None

    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                # 在图上画出一个矩形点
                cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 0, 255), -1)
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x-1, y-1), (x+1, y+1), (0, 255, 0), -1)

    def clear_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)

    def get_overlay(self):
        if self.current_overlay == self.seeds:
            return self.seed_overlay
        else:
            return self.segment_overlay

    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            # addWeighted--将两张图片叠加 0.9是图1的叠加比例，0.4是图二的叠加比例， 0.1是bias
            # dst = src1*alpha + src2*beta + gamma
            return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.image, 0.9, self.segment_overlay, 0.9, 0.1)

    def create_graph(self):
        if len(self.background_seeds) == 0 or len(self.foreground_seeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return

        print("Making graph")
        print("Finding foreground and background averages")
        self.find_averages()

        print("Cutting graph")
        self.cut_graph()

    def find_averages(self):
        foreground_points = np.zeros((len(self.foreground_seeds), 3))
        background_points = np.zeros((len(self.background_seeds), 3))

        for i, coordinate in enumerate(self.foreground_seeds):
            foreground_points[i, :] = self.image[coordinate[1], coordinate[0]]

        for i, coordinate in enumerate(self.background_seeds):
            background_points[i, :] = self.image[coordinate[1], coordinate[0]]

        clf1 = KMeans(n_clusters=15)
        clf1.fit(foreground_points)
        self.f_c = clf1.cluster_centers_


        clf2 = KMeans(n_clusters=10)
        clf2.fit(background_points)
        self.b_c = clf2.cluster_centers_


    def cut_graph(self):
        d_f = np.zeros((self.height, self.width))
        d_b = np.zeros((self.height, self.width))

        for i in range(self.height):
            for j in range(self.width):
                points_f = np.reshape(self.image[i, j, :], [1, 3])
                dist_sqrt_f = np.sqrt(np.sum((self.f_c - points_f) ** 2))
                d_f[i, j] = np.min(dist_sqrt_f)

        for i in range(self.height):
            for j in range(self.width):
                points_b = np.reshape(self.image[i, j, :], [1, 3])
                dist_sqrt_b = np.sqrt(np.sum((self.b_c - points_b) ** 2))
                d_b[i, j] = np.min(dist_sqrt_b)

        print(d_f)
        print(d_b)

        f = d_f - d_b
        tau = 0.1
        sigma = 0.05
        mu = 1000

        imgray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        u = primal_dual(imgray, sigma, tau, mu, f, iters=10)
        u[u > 0.5] = 1
        u[u <= 0.5] = 0

        for coordinate in self.background_seeds:
            u[coordinate[1], coordinate[0]] = 0

        for coordinate in self.foreground_seeds:
            u[coordinate[1], coordinate[0]] = 1

        u = np.array(u, dtype=np.int32)

        self.segment_overlay = np.zeros_like(self.segment_overlay)
        self.mask = np.zeros_like(self.image, dtype=bool)

        indices = np.where(u == 1)
        foreground_nodes_num = np.shape(indices)[1]
        for index in range(foreground_nodes_num):
            self.segment_overlay[indices[0][index], indices[1][index]] = (255, 255, 255)
            self.mask[indices[0][index], indices[1][index]] = (True, True, True)

    def save_image(self, filename):
        if self.mask is None:
            print('Please segment the image before saving.')
            return
        to_save = np.zeros_like(self.image)

        np.copyto(to_save, self.image, where=self.mask)
        cv2.imwrite(str(filename), self.segment_overlay)