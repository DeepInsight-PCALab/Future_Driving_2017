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



img_root = '/data3/XJTU2017/task_1/1495485141585513338/'
key      = '1495485141585513338'
img_json = '/data3/XJTU2017/task_1/Tusimple/label_data_0601.json'
output_root = './test_tusimple_imgs/'


def generate_labeled_images(img_root, img_json, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    total_cnt = 0
    #t = []
    with open(img_json) as data_file:
        for line in data_file:
            anno = json.loads(line)
            #t.append(anno)
            raw = anno['raw_file'] 
            if key not in raw:
                continue
            print anno
            total_cnt += 1
            if total_cnt > 3:
                exit()
            
            img_name = raw.split('/')[-1]
            img_path = os.path.join(img_root, img_name)           
            img_to   = os.path.join(output_root, img_name)
            
            y_list = anno['h_samples']
            x_matrix = anno['lanes']
            total_li = []
            for line in x_matrix:
                li = []
                for i in xrange(len(line)):
                    if line[i] > 0:
                        li.append((line[i], y_list[i]))
                total_li.append(li)

            img      = cv2.imread(img_path)
            print('img = ', img)
            print('type img = ', type(img))
            print('img.shape = ', (img).shape)
            exit()
            #new_img  = image_draw_line_list(img, total_li, (0, 0, 255), 3)
            #new_img  = image_draw_dot_line_list(img, total_li, (0, 0, 255), 3)
            new_img  = image_draw_half_line_list(img, total_li, (0, 0, 255), 2)
            h, w, c  = new_img.shape
            new_img  = cv2.resize(new_img, (w / 2, h / 2))
            cv2.imwrite(img_to, new_img)
            print('writing img ', img_to, 'with ', len(x_matrix), 'lines')

'''
def generate_labeled_images(img_root, img_xml, output_root):
    domtree = xml.dom.minidom.parse(img_xml)
    anno    = domtree.documentElement

    sons = os.listdir(img_root)
    for son in sons:
        img_path = os.path.join(img_root, son)
        img_id   = img_path.split('-')[-1].split('.')[0]

        target   = 'Frame%sTargetNumber' % img_id
        num = anno.getElementsByTagName(target)
        print num
        num = num[0].data
        print num

'''
generate_labeled_images(img_root, img_json, output_root)





