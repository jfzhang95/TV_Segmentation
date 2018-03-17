# -*- coding: utf-8 -*-
import cv2
import numpy as np
from utils import *

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
        self.load_image('images/cup1.jpg')
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
        self.background_average = np.zeros(3)
        self.foreground_average = np.zeros(3)

        for coordinate in self.background_seeds:
            print(self.image[coordinate[1], coordinate[0]])
            self.background_average += self.image[coordinate[1], coordinate[0]]

        self.background_average /= len(self.background_seeds)

        for coordinate in self.foreground_seeds:
            self.foreground_average += self.image[coordinate[1], coordinate[0]]

        self.foreground_average /= len(self.foreground_seeds)

    def cut_graph(self):
        self.foreground_plane = np.zeros((self.height, self.width, 3))
        self.background_plane = np.zeros((self.height, self.width, 3))

        self.foreground_plane[:, :, 0] = self.foreground_average[0]
        self.foreground_plane[:, :, 1] = self.foreground_average[1]
        self.foreground_plane[:, :, 2] = self.foreground_average[2]

        self.background_plane[:, :, 0] = self.background_average[0]
        self.background_plane[:, :, 1] = self.background_average[1]
        self.background_plane[:, :, 2] = self.background_average[2]

        d1 = self.image - self.foreground_plane
        f1 = np.sqrt(d1[:, :, 0] ** 2 + d1[:, :, 1] ** 2 + d1[:, :, 2] ** 2)

        d2 = self.image - self.background_plane
        f2 = np.sqrt(d2[:, :, 0] ** 2 + d2[:, :, 1] ** 2 + d2[:, :, 2] ** 2)
        f = f1 - f2
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