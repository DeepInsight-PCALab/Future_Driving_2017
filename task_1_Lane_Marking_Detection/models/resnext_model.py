#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch
import torch.optim as optim
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
from .resnext import *
import torch.nn.functional as F
import sys
sys.path.append('..')
sys.path.append('../..')
from common import image_draw_line_list, image_draw_dot_line_list, image_draw_half_line_list
import random


class ResNeXtModel(BaseModel):
    def __init__(self, opt):
        super(ResNeXtModel, self).__init__(opt)
        
        nb    = opt.batchSize
        sizeH, sizeW = opt.fineH, opt.fineW
        self.lam = opt.lam
        self.input_          = self.Tensor(nb, opt.input_nc, sizeH, sizeW)
        self.label_cls_      = self.Tensor(nb, 1, sizeH, sizeW)
        self.label_cls_mask_ = self.Tensor(nb, 1, sizeH, sizeW)
        self.label_up_       = self.Tensor(nb, opt.slicing + 1, sizeH, sizeW)
        self.label_up_mask_  = self.Tensor(nb, opt.slicing + 1, sizeH, sizeW)
        self.label_down_     = self.Tensor(nb, opt.slicing + 1, sizeH, sizeW)
        self.label_down_mask_= self.Tensor(nb, opt.slicing + 1, sizeH, sizeW)
        self.criterion       = torch.nn.SmoothL1Loss()

        #def resnext101(baseWidth, cardinality, slicing):
        self.net   = resnext101(4, 32, opt.slicing)
        if self.gpu_ids: 
            self.net   = torch.nn.DataParallel(self.net).cuda()

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.net, 'resnext101', which_epoch)

        if self.isTrain:
            self.optimizer = optim.SGD(self.net.parameters(), lr = self.opt.lr, momentum = 0.9, weight_decay = 5e-4)

    def set_input(self, input):
        self.input_.resize_(input.size()).copy_(input)

    def forward_input(self):
        self.input          = Variable(self.input_)
        self.pre_cls, self.pre_up, self.pre_down = self.net(self.input)
        return self.pre_cls, self.pre_up, self.pre_down

    
    def set_input_and_label(self, dic):
        #if self.opt.debug:
        #    print('dic = ', dic)
        input             = dic['input']
        label_cls         = dic['cls']
        label_cls_mask    = dic['cls_mask']
        label_up          = dic['up']
        label_up_mask     = dic['up_mask']
        label_down        = dic['down']
        label_down_mask   = dic['down_mask']

        self.input_.resize_(input.size()).copy_(input)
        self.label_cls_.resize_(label_cls.size()).copy_(label_cls)
        self.label_cls_mask_.resize_(label_cls_mask.size()).copy_(label_cls_mask)
        self.label_up_.resize_(label_up.size()).copy_(label_up)
        self.label_up_mask_.resize_(label_up_mask.size()).copy_(label_up_mask)
        self.label_down_.resize_(label_down.size()).copy_(label_down)
        self.label_down_mask_.resize_(label_down_mask.size()).copy_(label_down_mask)
        #if self.opt.debug:
        #    print('self.input_ = ', self.input_)

    def forward(self):
        self.input          = Variable(self.input_)
        self.label_cls      = Variable(self.label_cls_)
        self.label_cls_mask = Variable(self.label_cls_mask_)
        self.label_up       = Variable(self.label_up_)
        self.label_up_mask  = Variable(self.label_up_mask_)
        self.label_down     = Variable(self.label_down_)
        self.label_down_mask= Variable(self.label_down_mask_)

        self.pre_cls, self.pre_up, self.pre_down = self.net(self.input)
        self.loss_cls  = F.binary_cross_entropy(self.pre_cls, self.label_cls, weight = self.label_cls_mask * self.lam)
        self.loss_up   = self.criterion(self.pre_up   * self.label_up_mask,   self.label_up * self.label_up_mask)
        self.loss_down = self.criterion(self.pre_down * self.label_down_mask, self.label_down * self.label_down_mask)

    #def test(self): pass

    def optimize_parameters(self):
        self.forward()

        self.optimizer.zero_grad()
        self.loss      = self.loss_cls + self.loss_up + self.loss_down
        self.loss.backward()
        self.optimizer.step()

    def get_current_errors(self):
        loss_cls = self.loss_cls.data[0]
        loss_up  = self.loss_up.data[0]
        loss_down= self.loss_down.data[0]
        return OrderedDict([('loss_cls', loss_cls), ('loss_up', loss_up), ('loss_down', loss_down)])

    def isout(self, p):
        x = p[0]
        y = p[1]
        if not (0 <= x and x < self.opt.fineW and 0 <= y and y < self.opt.fineH):
            return True
        return False

    def decode_lanes(self, cls, up, down, threshold, debug = 0):
        fea_step_y   = self.opt.fineH / self.opt.feaH
        fea_step_x   = self.opt.fineW / self.opt.feaW
        slice_step_y = self.opt.fineH / self.opt.slicing

        lanes = []
        for h in xrange(self.opt.feaH):
            stdy = h * fea_step_y
            for w in xrange(self.opt.feaW):
                stdx = w * fea_step_x + 0.5 * fea_step_x
                prob = cls[0][h][w]
                if prob < threshold:
                    continue
                lane = []

                # up lane
                up_len  = int(up[0][h][w])
                y_id    = 1
                y_start = stdy
                while True:
                    pt = (stdx + up[y_id][h][w], y_start)
                    lane.append(pt)
                    y_id    += 1
                    y_start -= slice_step_y

                    if y_id >= up_len or y_id >= self.opt.slicing + 1:
                        break

                #if len(lane) >= 2:
                #    for i in xrange(3):
                #        pt = (2 * lane[-1][0] - lane[-2][0], 2 * lane[-1][1] - lane[-2][1])
                #        lane.append(pt)

                lane.reverse()
                # down lane
                down_len = int(down[0][h][w])
                y_id     = 1
                y_start  = stdy + slice_step_y
                while True:
                    pt = (stdx + down[y_id][h][w], y_start)
                    lane.append(pt)
                    y_id    += 1
                    y_start += slice_step_y

                    if (y_id >= down_len) or y_id >= self.opt.slicing + 1:
                        break
                # append
                #while True:
                #    if self.isout(lane[-1]):
                #        break
                #for i in xrange(10):
                #    pt = (2 * lane[-1][0] - lane[-2][0], 2 * lane[-1][1] - lane[-2][1])
                #    lane.append(pt)

                if debug == 1 and up_len <= 2 and down_len <= 2:
                    print('up_len = ', up_len, 'down_len = ', down_len, 'prob = ', prob)
                lanes.append((lane, prob))

        return lanes

    def todic(self, l):
        d = {}
        for v in l:
            d[v[1]] = v[0]
        return d

    def dis(self, la, lb):
        da = self.todic(la)
        db = self.todic(lb)
        cy = list(set(da.keys()) & set(db.keys()))
        sum = 0.0
        sumn = 0
        for y in cy:
            sum  += abs(da[y] - db[y])
            sumn += 1
        if sumn == 0:
            return float('inf')
        return sum / sumn
    
    def nms(self, lanes, dis_thres):
        las  = sorted(lanes, key = lambda x: x[1], reverse = True)
        le   = len(las)
        flag = np.zeros(le)
        res = []
        for i in xrange(le):
            if flag[i] == 1:
                continue

            la = las[i][0]
            res.append(la)
            for j in xrange(i + 1, le):
                if flag[j] == 1:
                    continue
                if self.dis(la, las[j][0]) < dis_thres:
                    flag[j] = 1
        return res

    def draw_lanes(self, lanes):
        img = util.tensor2im(self.input.data[0])
        img = image_draw_line_list(img, lanes)
        return img

    def draw(self, cls, up, down, threshold):
        img = util.tensor2im(self.input.data[0])
        # TODO uncover the label into lanes
        lanes = self.decode_lanes(cls, up, down, threshold)
        lanes = self.nms(lanes, 10)
        img = image_draw_line_list(img, lanes)
        return img

    def draw_one(self, cls, up, down, threshold):
        img = util.tensor2im(self.input.data[0])
        # TODO uncover the label into lanes
        lanes = self.decode_lanes(cls, up, down, threshold, debug = 1)
        lanes = self.nms(lanes, 10)
        le = len(lanes)
        one_lanes = []
        for i in xrange(le):
            if len(lanes[i]) < 4:
                one_lanes.append(lanes[i])
                break
        img = image_draw_line_list(img, one_lanes)
        return img

    def get_current_visuals(self, threshold):
        imgA = self.draw(self.pre_cls.data[0], self.pre_up.data[0], self.pre_down.data[0], threshold)
        imgB = self.draw(self.label_cls.data[0], self.label_up.data[0], self.label_down.data[0], threshold)
        return OrderedDict([('img_predict', imgA), ('img_label', imgB)])
        
    def save(self, label):
        self.save_network(self.net, 'resnext101', label, self.gpu_ids)


