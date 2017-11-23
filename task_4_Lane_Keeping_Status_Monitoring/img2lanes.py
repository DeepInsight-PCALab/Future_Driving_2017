#!/usr/bin/env python
# coding=utf-8
def postdeal(lanes, H, W):
    new_lanes = []
    for lane_group in lanes:
        new_lane = []
        for p in lane_group[0]:
            x = p[0]
            y = p[1]
            new_lane.append((x, y))
            if not (0 <= x and x < W and 0 <= y and y < H):
                break
        #new_lane.reverse()
        lane_group[0] = new_lane
        new_lanes.append(lane_group)
    return new_lanes

import sys
#sys.path.append('..')
sys.path.append('../task_1_Lane_Marking_Detection/')
from options.test_xml_options import TestXMLOptions
from models.models import create_model
from data.transfer import Transformer
import cv2
from common import image_draw_line_cls_list

class getLanes(object):
    def __init__(self):
        opt = TestXMLOptions().parse()
        opt.checkpoints = './checkpoints/'
        opt.name = 'task_1'
        self.model = create_model(opt)
        self.trans = Transformer(opt)

    def get_lanes(self, img_path, cls_thres = 0.9, nms_dis = 10):

        img = self.trans.gao_image_path(img_path)
        imgW, imgH = self.trans.original_width, self.trans.original_height

        img = img.unsqueeze(0)
        self.model.set_input(img)
        cls, up, down, color, type = self.model.forward_input()

        lanes = self.model.decode_lanes(cls.data[0], up.data[0], down.data[0], color.data[0], type.data[0], cls_thres)
        lanes = self.model.nms(lanes, nms_dis)
        lanes = self.model.scale(lanes, imgH, imgW)
        lanes = postdeal(lanes, imgH, imgW)

        '''
        old_img = cv2.imread(img_path)
        new_img = image_draw_line_cls_list(old_img, lanes, (0, 0, 255), thin = 5)
        new_img_path = '_Check.jpg'
        cv2.imwrite(new_img_path, new_img)
        '''

        res = []
        for lane in lanes:
            res.append(lane[0])

        return res

def get_lanes(img_path, cls_thres = 0.9, nms_dis = 10):
    opt = TestXMLOptions().parse()
    opt.checkpoints = './checkpoints/'
    opt.name = 'task_1'
    model = create_model(opt)
    trans = Transformer(opt)
    img = trans.gao_image_path(img_path)
    imgW, imgH = trans.original_width, trans.original_height

    img = img.unsqueeze(0)
    model.set_input(img)
    cls, up, down, color, type = model.forward_input()

    lanes = model.decode_lanes(cls.data[0], up.data[0], down.data[0], color.data[0], type.data[0], cls_thres)
    lanes = model.nms(lanes, nms_dis)
    lanes = model.scale(lanes, imgH, imgW)
    lanes = postdeal(lanes, imgH, imgW)

    old_img = cv2.imread(img_path)
    new_img = image_draw_line_cls_list(old_img, lanes, (0, 0, 255), thin = 5)
    new_img_path = '_Check.jpg'
    cv2.imwrite(new_img_path, new_img)

    res = []
    for lane in lanes:
        res.append(lane[0])

    return res
# lanes = get_lanes('/data3/XJTU2017/task_4/TSD-LKSM/TSD-LKSM-00121/TSD-LKSM-00121-00000.png')
# print(lanes)
