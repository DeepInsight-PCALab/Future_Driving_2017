#!/usr/bin/env python
# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../..')
from common import load_annotation, image_draw_line_list, image_draw_dot_line_list, image_draw_half_line_list
import os
import xml.dom.minidom
import cv2
import json
from pprint import pprint

img_root = '/data3/DeepInsight/Lane_PAMI/Datasets/tolabel/'
img_txt  = '/data3/DeepInsight/Lane_PAMI/Datasets/lane_label/'
output_root = '/data3/XJTU2017/task_1/PAMI_tolabel/'


def generate_labeled_images(img_root, img_txt, output_root, img_file, pid):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    #img_files = os.listdir(img_root)
    #for img_file in img_files:
    img_file_path   = os.path.join(img_root, img_file)
    label_file_path = os.path.join(img_txt, img_file + '_L')

    output_img_file = os.path.join(output_root, img_file + '_LR')
    if not os.path.exists(output_img_file):
        os.makedirs(output_img_file)

    img_names = os.listdir(img_file_path)
    for img_name in img_names:
        img_path = os.path.join(img_file_path, img_name)
        label_path = os.path.join(label_file_path, img_name.split('.')[0] + '.txt')
        #print img_path
        #print label_path
        if not (os.path.exists(img_path) and os.path.exists(label_path)):
            continue

        img   = cv2.imread(img_path)
        label = open(label_path).readlines()
        for id, line in enumerate(label):
            img_to   = os.path.join(output_img_file, img_name.replace('.', '_%02d.' % id))
            arr = line.split(' ')
            lanes = []
            lane = []
            for i in xrange(0, len(arr), 2):
                x = float(arr[i])
                y = float(arr[i + 1])
                if x == 0 and y == 0:
                    break
                lane.append([x, y])
                #lane.append([float(arr[i]), float(arr[i + 1])])
            lanes.append(lane)

            new_img  = image_draw_line_list(img, lanes, (0, 0, 255), 2)
            h, w, c  = new_img.shape
            new_img  = cv2.resize(new_img, (w / 2, h / 2))
            cv2.imwrite(img_to, new_img)
            print 'pid = ', pid, img_to

import multiprocessing
class gao(multiprocessing.Process):
    def __init__(self, img_file, pid):
        multiprocessing.Process.__init__(self)
        self.img_file = img_file
        self.lxid      = pid

    def run(self):
        generate_labeled_images(img_root, img_txt, output_root, self.img_file, self.lxid)

img_files = os.listdir(img_root)
for pid, img_file in enumerate(img_files):
    p = gao(img_file, pid)
    p.start()
    #p.join()
