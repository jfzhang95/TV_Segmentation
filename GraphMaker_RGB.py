import numpy as np
import cv2
from numpy import *
from utils import *
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage import img_as_ubyte
from dijkstar import Graph, find_path
from numba import jit
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

    def __init__(self):
        self.depth = None
        self.image = None
        self.superpixel_image = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None

        self.load_image('images/cup1.jpg')

        self.background_seeds = []
        self.foreground_seeds = []
        self.foreground_superseeds = []
        self.background_superseeds = []
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds

        self.ave_RGB = None
        self.superpixel_segment = None
        self.super_edge = None

        self.RGB_map = None
        self.cfd_RGB= None

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
        if len(self.foreground_seeds) == 0:
            print("Please enter at least one foreground seed.")
            return

        starttime = datetime.datetime.now()

        if self.flag == True:
            print("Making graph")

            #########生成超像素#########
            self.getSuperpixel()
            #########构建map###########
            self.getSuperpixelValue()
            self.flag = False

        #########获取置信图#########
        self.getLocalTerm()
        #########Cut Graph########
        self.cut_graph()

        endtime = datetime.datetime.now()
        print("Complete! Total: " + str((endtime - starttime).seconds))

    # @njit
    def getSuperpixel(self):
        starttime = datetime.datetime.now()
        self.superpixel_segment = self.get_superpixel()
        endtime = datetime.datetime.now()
        print("get superpixel: " + str((endtime - starttime).seconds))
        # init super-pixel
        self.n_seg = np.amax(self.superpixel_segment) + 1  # the number of super-pixels
        print("the number of superpixel:", self.n_seg)
        self.num_seg = np.zeros(self.n_seg, dtype=int)  # count for each super-pixel

        # Go through all pixels in image
        for x in range(0, self.height):
            for y in range(0, self.width):
                self.num_seg[self.superpixel_segment[x, y]] += 1

        # Computing edge link relations
        # TODO: improve method to accelerate algorithms, numba
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

        ########check edge on superpixels#########
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
        # cv2.imwrite("./results/EdgeImg.jpg", temp_img)

    @jit
    def getSuperpixelValue(self):

        # TODO: accelerate

        starttime = datetime.datetime.now()
        self.ave_RGB = np.zeros((self.n_seg, 3), dtype=np.float)
        for x in range(0, self.height):
            for y in range(0, self.width):
                self.ave_RGB[self.superpixel_segment[x, y]] += self.image[x, y]
        for i in range(0, self.n_seg):
            self.ave_RGB[i] /= self.num_seg[i]
        endtime = datetime.datetime.now()
        print("get superpixel value for each cue: " + str((endtime - starttime).seconds))


    # @jit
    def getLocalTerm(self):
        for coordinate in self.foreground_seeds:
            if self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1] not in self.foreground_superseeds:
                self.foreground_superseeds.append(self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1])
        for coordinate in self.background_seeds:
            if self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1] not in self.background_superseeds:
                self.background_superseeds.append(self.superpixel_segment[coordinate[1] - 1, coordinate[0] - 1])


        cost_func = lambda u, v, e, prev_e: e['cost']

        #######For RGB foreground#######
        G_RGB = Graph()

        # Build graph, compute threshold
        weight = [[] for i in range(0, self.n_seg)]
        aveMinWeight = 0
        for u in range(0, self.n_seg):
            minWeight = self.MAXIMUM
            for v in self.super_edge[u]:
                w = self.eu_dis(self.ave_RGB[u], self.ave_RGB[v])
                weight[u].append((v, w))
                if minWeight > w:
                    minWeight = w
            aveMinWeight += minWeight
        aveMinWeight /= self.n_seg

        for u in range(0, self.n_seg):
            for v, w in weight[u]:
                if w < aveMinWeight:
                    G_RGB.add_edge(u, v, {'cost': 0})
                else:
                    G_RGB.add_edge(u, v, {'cost': w})

        for v in self.foreground_superseeds:
            G_RGB.add_edge(v, 's', {'cost': 0})

        starttime = datetime.datetime.now()
        RGB_disFore = np.zeros(self.n_seg, dtype=float)
        RGB_disFore.fill(self.MAXIMUM)

        # for fg in self.foreground_superseeds:
        for v in range(0, self.n_seg):
            info = find_path(G_RGB, v, 's', cost_func=cost_func)
            RGB_disFore[v] = info.total_cost

        endtime = datetime.datetime.now()
        print("compute RGB disfore: " + str((endtime - starttime).seconds))

        #######get background_seed#######
        #######NEED TO BE IMPROVED#######
        #######When the object is very big, this program will cause problem######
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
            # 150 is a threshold(hyper-parameters), which need to be chosen carefully!
            if RGB_disFore[boundary] > 150 and boundary not in self.background_superseeds:
                self.background_superseeds.append(boundary)

        tmp_img = self.image.copy()
        for x in range(0, self.height):
            for y in range(0, self.width):
                if self.superpixel_segment[x, y] in self.background_superseeds:
                    tmp_img[x, y] = (0, 0, 255)
                if self.superpixel_segment[x, y] in self.foreground_superseeds:
                    tmp_img[x, y] = (0, 255, 0)
        cv2.imwrite("./results/backseeds.jpg", tmp_img)

        for v in self.background_superseeds:
            G_RGB.add_edge(v, 't', {'cost': 0})

        #######For LAB background#######
        starttime = datetime.datetime.now()

        RGB_disBack = np.zeros(self.n_seg, dtype=float)
        RGB_disBack.fill(self.MAXIMUM)
        for v in range(0, self.n_seg):
            info = find_path(G_RGB, v, 't', cost_func=cost_func)
            RGB_disBack[v] = info.total_cost

        endtime = datetime.datetime.now()
        print("compute RGB disback: " + str((endtime - starttime).seconds))

        self.cfd_RGB = np.zeros(self.n_seg, dtype=float)

        self.cfd_RGB = RGB_disFore - RGB_disBack

        self.f = np.zeros_like(self.superpixel_segment)
        for i in range(0, self.n_seg):
            self.f[self.superpixel_segment == i] = self.cfd_RGB[i]

        f = self.f.copy()
        f[f <= 0] = 0
        f[f > 0] = 255
        cv2.imwrite('./results/SegImg.jpg', f)


    # def construct_confident(self):
    #     self.segment_overlay = np.zeros_like(self.segment_overlay)


    @staticmethod
    def eu_dis(v1, v2):
        return np.sqrt(np.sum((v1-v2) ** 2))


    def cut_graph(self):
        tau = 0.1
        sigma = 0.05
        mu = 10000

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
            self.segment_overlay[indices[0][index], indices[1][index]] = (255, 0, 255)
            self.mask[indices[0][index], indices[1][index]] = (True, True, True)


    # SLIC SuperPixel
    def get_superpixel(self):
        segments = slic(self.image, n_segments=1500, compactness=10)
        self.superpixel_image = img_as_ubyte(mark_boundaries(self.image, segments))
        return segments


    def save_image(self, filename):
        if self.mask is None:
            print('Please segment the image before saving.')
            return
        # to_save = np.zeros_like(self.image)
        # np.copyto(to_save, self.image, where=self.mask)

        cv2.imwrite(str(filename), self.segment_overlay)