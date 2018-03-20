import numpy as np
import cv2
from numpy import *
from utils import *
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
from dijkstar import Graph, find_path

import datetime


class GraphMaker:
    foreground = 1
    background = 0

    seeds = 0
    segmented = 1
    flag = True

    default = 0.5
    MAXIMUM = 1000000000


    ns = 1
    ###各种参数###
    lamda = 0.1
    sigma1 = 0.5
    sigma2 = 4000
    sigma3 = 15

    def __init__(self):
        self.depth = None
        self.image = None
        self.superpixel_image = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None

        self.load_image('images/rgb.jpg')

        self.background_seeds = []
        self.foreground_seeds = []
        self.foreground_superseeds = []
        self.background_superseeds = []
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds

        self.ave_LAB = None
        self.superpixel_segment = None
        self.super_edge = None

        self.LAB_map = None
        self.cfd_LAB = None

    def load_image(self, filename):
        self.image = cv2.imread(filename)
        self.height, self.width, _ = np.shape(self.image)
        print(self.height, self.width)
        self.superpixel_image = self.image.copy()
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)

    def add_seed(self, x, y, type):
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 255), -1)
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (x - 1, y - 1), (x + 1, y + 1), (0, 255, 0), -1)

    def clear_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)

    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.image, 0.9, self.segment_overlay, 0.9, 0.1)

    def create_graph(self):
        starttime = datetime.datetime.now()

        if self.flag == True:
            print("Making graph")

            #########生成超像素#########
            self.getSuperpixel()
            #########构建map###########
            self.getCueValue()
            self.flag = False

        #########获取置信图#########
        self.getConfidentMap()


        if len(self.background_superseeds) == 0 or len(self.foreground_superseeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return

        self.cut_graph()
        endtime = datetime.datetime.now()
        print("Complete! Total: " + str((endtime - starttime).seconds))


    def getSuperpixel(self):
        starttime = datetime.datetime.now()
        self.superpixel_segment = self.get_superpixel()
        endtime = datetime.datetime.now()
        print("get superpixel: " + str((endtime - starttime).seconds))
        # init super-pixel
        self.n_seg = np.amax(self.superpixel_segment) + 1  # the number of super-pixels
        print(self.n_seg)
        self.num_seg = np.zeros(self.n_seg, dtype=int)  # count for every cluster

        # Go through all pixels in image
        for x in range(0, self.height):
            for y in range(0, self.width):
                self.num_seg[self.superpixel_segment[x, y]] += 1

        # Computing edge link relations
        # TODO: improve method to accelerate algorithms
        starttime = datetime.datetime.now()
        self.super_edge = [[] for _ in range(0, self.n_seg)]
        for x in range(0, self.height):
            for y in range(0, self.width):
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if i == 0 and j == 0:
                            continue
                        if (x + i < 0 or x + i >= len(self.superpixel_segment) or y + j < 0 or y + j >= len(
                                self.superpixel_segment[0])):
                            continue
                        if (self.superpixel_segment[x, y] != self.superpixel_segment[x + i, y + j]):
                            if self.superpixel_segment[x, y] not in self.super_edge[
                                self.superpixel_segment[x + i, y + j]]:
                                self.super_edge[self.superpixel_segment[x + i, y + j]].append(
                                    self.superpixel_segment[x, y])
                                self.super_edge[self.superpixel_segment[x, y]].append(
                                    self.superpixel_segment[x + i, y + j])
        endtime = datetime.datetime.now()
        print("compute each edge: " + str((endtime - starttime).seconds))

        #########check edge on superpixels#########
        # temp_img = self.superpixel_image.copy()
        # ave_cor = [[0, 0] for i in range(0, self.n_seg)]
        # for x in range(0, self.height):
        #     for y in range(0, self.width):
        #         ave_cor[self.superpixel_segment[x, y]][0] += x
        #         ave_cor[self.superpixel_segment[x, y]][1] += y
        # for i in range(0, self.n_seg):
        #     ave_cor[i] /= self.num_seg[i]
        # for i in range(0, self.n_seg):
        #     for j in self.super_edge[i]:
        #         if j < i:
        #             continue
        #         cv2.line(temp_img, (int(ave_cor[i][1]), int(ave_cor[i][0])), (int(ave_cor[j][1]), int(ave_cor[j][0])),
        #                  (0, 0, 255), 1)
        # temp_img = temp_img.astype('uint8')
        # cv2.imwrite("./results/edgeImg.jpg", temp_img)

    def getCueValue(self):

        self.LAB_map = self.LABMap()

        starttime = datetime.datetime.now()
        # LABMap for superpixel
        self.ave_LAB = np.zeros((self.n_seg, 3), dtype=np.float)
        for x in range(0, self.height):
            for y in range(0, self.width):
                self.ave_LAB[self.superpixel_segment[x, y]] += self.LAB_map[x, y]
        for i in range(0, self.n_seg):
            self.ave_LAB[i] /= self.num_seg[i]
        endtime = datetime.datetime.now()
        print("get superpixel value for each cue: " + str((endtime - starttime).seconds))

    def getConfidentMap(self):
        # ====================================================================================
        #         我们采用简单有效的重量削减方法来解决这个问题，而不是探索复杂的功能和复杂的补丁外观距离度量。
        #         补丁外观距离简单地取为两个斑块的平均颜色（LAB颜色空间）之间的差异（归一化为[0,1]）。 对于
        #         每个补丁，我们选择其与所有邻居的最小外观距离，然后我们选择“无意义”距离阈值作为与所有补丁的
        #         所有这样最小距离的平均值。如果任何距离小于该阈值，则被认为是不显著的且被限制为0。这种内部
        #         边缘权重的计算是非常有效的，其有效性如图3所示。
        # ====================================================================================
        for coordinate in self.foreground_seeds:
            if self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1] not in self.foreground_superseeds:
                self.foreground_superseeds.append(self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1])
        for coordinate in self.background_seeds:
            if self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1] not in self.background_superseeds:
                self.background_superseeds.append(self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1])

        cost_func = lambda u, v, e, prev_e: e['cost']

        #######For LAB foreground#######

        G_LAB = Graph()

        # Build graph, compute threshold
        weight = [[] for i in range(0, self.n_seg)]
        aveMinWeight = 0
        for u in range(0, self.n_seg):
            minWeight = self.MAXIMUM
            for v in self.super_edge[u]:
                w = self.eu_dis(self.ave_LAB[u], self.ave_LAB[v])
                weight[u].append((v, w))
                if minWeight > w:
                    minWeight = w
            aveMinWeight += minWeight
        aveMinWeight /= self.n_seg

        for u in range(0, self.n_seg):
            for v, w in weight[u]:
                if w < aveMinWeight:
                    G_LAB.add_edge(u, v, {'cost': 0})
                else:
                    G_LAB.add_edge(u, v, {'cost': w})

        for v in self.foreground_superseeds:
            G_LAB.add_edge(v, 's', {'cost': 0})

        starttime = datetime.datetime.now()
        Lab_disFore = np.zeros(self.n_seg, dtype=float)
        Lab_disFore.fill(self.MAXIMUM)

        # for fg in self.foreground_superseeds:
        for v in range(0, self.n_seg):
            info = find_path(G_LAB, v, 's', cost_func=cost_func)
            Lab_disFore[v] = info.total_cost

        endtime = datetime.datetime.now()
        print("compute LAB disfore: " + str((endtime - starttime).seconds))

        #######get background_seed#######
        #######NEED TO BE IMPROVED#######
        #######When object is very big, this program will cause problem######
        boundary_superpixel = []
        for x in range(0, self.height):
            if self.superpixel_segment[x, 0] not in boundary_superpixel:
                boundary_superpixel.append(self.superpixel_segment[x, 0])
            if self.superpixel_segment[x, self.width - 1] not in boundary_superpixel:
                boundary_superpixel.append(self.superpixel_segment[x, self.width - 1])
        for y in range(0, self.width):
            if self.superpixel_segment[0, y] not in boundary_superpixel:
                boundary_superpixel.append(self.superpixel_segment[0, y])
            if self.superpixel_segment[self.height - 1, y] not in boundary_superpixel:
                boundary_superpixel.append(self.superpixel_segment[self.height - 1, y])
        for boundary in boundary_superpixel:
            # 0.1 is a threshold, which need to be chosen carefully!
            if Lab_disFore[boundary] > 0.1 and boundary not in self.background_superseeds:
                self.background_superseeds.append(boundary)

        for v in self.background_superseeds:
            G_LAB.add_edge(v, 't', {'cost': 0})

        #######For LAB background#######
        starttime = datetime.datetime.now()

        Lab_disBack = np.zeros(self.n_seg, dtype=float)
        Lab_disBack.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_LAB, v, 't', cost_func=cost_func)
            Lab_disBack[v] = info.total_cost

        ######I think it is no need to do normalization#####
        # Maxdis = 0
        # for i in range(0, self.n_seg):
        #     if Maxdis < Lab_disBack[i]:
        #         Maxdis = Lab_disBack[i]
        # for i in range(0, self.n_seg):
        #     Lab_disBack[i] /= Maxdis

        endtime = datetime.datetime.now()
        print("compute LAB disback: " + str((endtime - starttime).seconds))

        self.cfd_LAB = np.zeros(self.n_seg, dtype=float)

        self.cfd_LAB = Lab_disFore - Lab_disBack

        self.f = np.zeros_like(self.superpixel_segment)
        for i in range(0, self.n_seg):
            self.f[self.superpixel_segment == i] = self.cfd_LAB[i]

    def construct_confident(self):

        self.segment_overlay = np.zeros_like(self.segment_overlay)

    @staticmethod
    def eu_dis(v1, v2):
        return np.sqrt((v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2)

    def cut_graph(self):
        tau = 0.1
        sigma = 0.05
        mu = 100

        imgray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        u = primal_dual(imgray, sigma, tau, mu, self.f, iters=50)
        u[u > 0.5] = 1
        u[u <= 0.5] = 0

        for coordinate in self.background_seeds:
            u[coordinate[1], coordinate[0]] = 0

        for coordinate in self.foreground_seeds:
            u[coordinate[1], coordinate[0]] = 1

        u = np.array(u, dtype=np.int16)

        self.segment_overlay = np.zeros_like(self.segment_overlay)
        self.mask = np.zeros_like(self.image, dtype=bool)

        indices = np.where(u == 1)
        foreground_nodes_num = np.shape(indices)[1]
        for index in range(foreground_nodes_num):
            self.segment_overlay[indices[0][index], indices[1][index]] = (255, 255, 255)
            self.mask[indices[0][index], indices[1][index]] = (True, True, True)

    # SLIC SuperPixel
    def get_superpixel(self):
        segments = slic(self.image, n_segments=2000)
        self.superpixel_image = img_as_ubyte(mark_boundaries(self.image, segments))
        return segments

    # construct LABMap
    def LABMap(self):
        LAB_map = cv2.cvtColor(self.image, cv2.COLOR_RGB2Lab)
        # LAB_map = np.zeros_like(LAB_map_raw, dtype=np.int8)
        # for i in range(len(LAB_map)):
        #     for j in range(len(LAB_map[0])):
        #         LAB_map[i, j][0] = LAB_map_raw[i, j][0] / 255 * 100
        #         LAB_map[i, j][1] = LAB_map_raw[i, j][1] - 128
        #         LAB_map[i, j][2] = LAB_map_raw[i, j][2] - 128
        return LAB_map

    def save_image(self, filename):
        if self.mask is None:
            print('Please segment the image before saving.')
            return
        to_save = np.zeros_like(self.image)

        np.copyto(to_save, self.image, where=self.mask)
        cv2.imwrite(str(filename), self.segment_overlay)