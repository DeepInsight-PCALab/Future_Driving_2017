#!/usr/bin/env python
# coding=utf-8

import math
def postdeal(lanes, H, W):
    def isin(p):
        x, y = p[0], p[1]
        if (0 <= x and x < W and 0 <= y and y < H):
            return True
        return False

    def dis(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.sqrt(dx * dx + dy * dy)

    new_lanes = []
    for lane_group in lanes:
        new_lane = []
        for p in lane_group[0]:
            x = int(p[0])
            y = int(p[1])
            if not (0 <= x and x < W and 0 <= y and y < H):
                if len(new_lane) > 0:
                    l = new_lane[-1]
                    r = [x, y]
                    eps = 1e-4
                    while dis(l, r) > eps:
                        m = [(l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0]
                        if isin(m):
                            l = m
                        else:
                            r = m
                    new_lane.append([int(l[0]), int(l[1])])
                break
            new_lane.append([x, y])
        new_lane.reverse()
        lane_group[0] = new_lane
        new_lanes.append(lane_group)
    return new_lanes

import time
import os
import cv2
from options.test_xml_options   import TestXMLOptions
print ('go')
from models.models          import create_model
print ('go')
from data.transfer          import Transformer
from util.util              import save_image
import sys
sys.path.append('..')
from common import XmlWriter, image_draw_line_cls_list


opt         = TestXMLOptions().parse()
model       = create_model(opt)
trans       = Transformer(opt)
save_dir    = opt.results_xml_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

st          = time.time()
folder_list = os.listdir(opt.test_dir)
total_c     = 0
for folder in folder_list:
    img_folder =  os.path.join(opt.test_dir, folder)
    if not os.path.isdir(img_folder):
        continue
    save_folder = os.path.join(save_dir, folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    img_list = os.listdir(img_folder)
    img_list = sorted(img_list)

    save_xml_dir  = os.path.join(save_dir, 'TSD-Lane-Result-PCALab')
    if not os.path.exists(save_xml_dir):
        os.makedirs(save_xml_dir)
    save_xml_path = os.path.join(save_xml_dir, folder + '-Result.xml')
    xml_writer   = XmlWriter(save_xml_path)

    one_st = time.time()
    for i, img_name in enumerate(img_list):
        img_path = os.path.join(img_folder, img_name)
        
        img = trans.gao_image_path(img_path)
        imgW, imgH = trans.original_width, trans.original_height

        img = img.unsqueeze(0)
        model.set_input(img)
        cls, up, down, color, type = model.forward_input()

        lanes = model.decode_lanes(cls.data[0], up.data[0], down.data[0], color.data[0], type.data[0], opt.cls_thres)
        lanes = model.nms(lanes, 20)

        # save img
        if opt.save_every_img == 1:
            lane_img = model.draw_lanes(lanes)
            save_img_path = os.path.join(save_folder, img_name)
            save_image(lane_img, save_img_path)

        # save xml obj
        lanes = model.scale(lanes, imgH, imgW)
        lanes = postdeal(lanes, imgH, imgW)
        xml_writer.add_lanes(lanes, i)
        total_c += 1

        if i == 0:
            old_img = cv2.imread(img_path)
            new_img = image_draw_line_cls_list(old_img, lanes, (0, 0, 255), thin = 5)
            new_save_dir = os.path.join(save_dir, 'Checks')
            new_img_path = os.path.join(new_save_dir, folder + '-Check.jpg')
            if not os.path.exists(new_save_dir):
                os.makedirs(new_save_dir)
            cv2.imwrite(new_img_path, new_img)

    xml_writer.write_xml()
    print('One %s folder time: %.2f min' %(folder, (time.time() - one_st) / 60.0))

total_t = (time.time() - st) / 60.0
print('Total time = %.2f min for %d samples' % (total_t, total_c))
