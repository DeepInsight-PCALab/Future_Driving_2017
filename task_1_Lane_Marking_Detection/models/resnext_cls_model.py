#!/usr/bin/env python
# coding=utf-8

import numpy as np
import torch
import torch.optim as optim
from collections import OrderedDict
from torch.autograd import Variable
#import itertools
import util.util as util
from .base_model import BaseModel
from .resnext import *
import torch.nn.functional as F
import sys
sys.path.append('..')
sys.path.append('../..')
#from common import image_draw_line_list, image_draw_dot_line_list, image_draw_half_line_list, image_draw_line_cls_list
from common import image_draw_line_cls_list


class ResNeXtClsModel(BaseModel):
    def __init__(self, opt):
        super(ResNeXtClsModel, self).__init__(opt)
        
        nb    = opt.batchSize
        sizeH, sizeW = opt.fineH, opt.fineW
        self.lam = opt.lam
        self.input_               = self.Tensor(nb, opt.input_nc, sizeH, sizeW)
        self.label_cls_           = self.Tensor(nb, 1, sizeH, sizeW)
        self.label_cls_mask_      = self.Tensor(nb, 1, sizeH, sizeW)
        self.label_up_            = self.Tensor(nb, opt.slicing + 1, sizeH, sizeW)
        self.label_up_mask_       = self.Tensor(nb, opt.slicing + 1, sizeH, sizeW)
        self.label_down_          = self.Tensor(nb, opt.slicing + 1, sizeH, sizeW)
        self.label_down_mask_     = self.Tensor(nb, opt.slicing + 1, sizeH, sizeW)
        # add these for more information
        self.label_color_         = self.Tensor(nb, 1, sizeH, sizeW)
        self.label_color_mask_    = self.Tensor(nb, 1, sizeH, sizeW)
        self.label_type_          = self.Tensor(nb, 1, sizeH, sizeW)
        self.label_type_mask_     = self.Tensor(nb, 1, sizeH, sizeW)

        self.criterion       = torch.nn.SmoothL1Loss()

        #def resnext101(baseWidth, cardinality, slicing):
        if opt.depth == 101:
            self.net   = resnext101(4, 32, opt.slicing)
        elif opt.depth == 152:
            self.net   = resnext152(4, 32, opt.slicing)


        if (not self.isTrain or opt.continue_train): #or (self.isTrain and not opt.pretrain):
            which_epoch = opt.which_epoch
            self.load_network(self.net, 'resnext%d' % (self.opt.depth), which_epoch)

        if (self.isTrain and opt.pretrain):
            print 'load pretrained model: ', opt.pretrain
            mydict  = self.net.state_dict()
            olddict = torch.load(opt.pretrain)
            mydict.update(olddict)
            print 'mydict more:', set(mydict.keys()) - set(olddict.keys())
            self.net.load_state_dict(mydict)

        if self.gpu_ids: 
            self.net   = torch.nn.DataParallel(self.net).cuda()

        if self.isTrain:
            if opt.finetune_cls == 1:
                print 'finetune cls layers: color and type!!!'
                self.optimizer = optim.SGD(list(self.net.module.layer_color.parameters()) + list(self.net.module.layer_type.parameters()), lr = self.opt.lr, momentum = 0.9, weight_decay = 5e-4)
            else:
                print 'train all layers!!!'
                self.optimizer = optim.SGD(self.net.parameters(), lr = self.opt.lr, momentum = 0.9, weight_decay = 5e-4)

    def set_input(self, input):
        self.input_.resize_(input.size()).copy_(input)

    def forward_input(self):
        self.input          = Variable(self.input_)
        self.pre_cls, self.pre_up, self.pre_down, self.pre_color, self.pre_type = self.net(self.input)
        return self.pre_cls, self.pre_up, self.pre_down, self.pre_color, self.pre_type

    
    def set_input_and_label(self, dic):
        input             = dic['input']
        label_cls         = dic['cls']
        label_cls_mask    = dic['cls_mask']
        label_up          = dic['up']
        label_up_mask     = dic['up_mask']
        label_down        = dic['down']
        label_down_mask   = dic['down_mask']
        label_color       = dic['color']
        label_color_mask  = dic['color_mask']
        label_type        = dic['type']
        label_type_mask   = dic['type_mask']

        self.input_.resize_(input.size()).copy_(input)
        self.label_cls_.resize_(label_cls.size()).copy_(label_cls)
        self.label_cls_mask_.resize_(label_cls_mask.size()).copy_(label_cls_mask)
        self.label_up_.resize_(label_up.size()).copy_(label_up)
        self.label_up_mask_.resize_(label_up_mask.size()).copy_(label_up_mask)
        self.label_down_.resize_(label_down.size()).copy_(label_down)
        self.label_down_mask_.resize_(label_down_mask.size()).copy_(label_down_mask)
        self.label_color_.resize_(label_color.size()).copy_(label_color)
        self.label_color_mask_.resize_(label_color_mask.size()).copy_(label_color_mask)
        self.label_type_.resize_(label_type.size()).copy_(label_type)
        self.label_type_mask_.resize_(label_type_mask.size()).copy_(label_type_mask)

    def forward(self):
        self.input          = Variable(self.input_)
        self.label_cls      = Variable(self.label_cls_)
        self.label_cls_mask = Variable(self.label_cls_mask_)
        self.label_up       = Variable(self.label_up_)
        self.label_up_mask  = Variable(self.label_up_mask_)
        self.label_down     = Variable(self.label_down_)
        self.label_down_mask= Variable(self.label_down_mask_)
        self.label_color     = Variable(self.label_color_)
        self.label_color_mask= Variable(self.label_color_mask_)
        self.label_type     = Variable(self.label_type_)
        self.label_type_mask= Variable(self.label_type_mask_)

        self.pre_cls, self.pre_up, self.pre_down, self.pre_color, self.pre_type = self.net(self.input)
        self.loss_cls   = F.binary_cross_entropy(self.pre_cls, self.label_cls, weight = self.label_cls_mask * self.lam)
        self.loss_up    = self.criterion(self.pre_up   * self.label_up_mask,   self.label_up * self.label_up_mask)
        self.loss_down  = self.criterion(self.pre_down * self.label_down_mask, self.label_down * self.label_down_mask)
        self.loss_color = F.binary_cross_entropy(self.pre_color, self.label_color, weight = self.label_color_mask * (self.lam))
        self.loss_type  = F.binary_cross_entropy(self.pre_type, self.label_type, weight = self.label_type_mask * (self.lam))


    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.loss      = self.loss_cls + self.loss_up + self.loss_down + self.loss_color + self.loss_type
        self.loss.backward()
        self.optimizer.step()

    def get_current_errors(self):
        loss_cls = self.loss_cls.data[0]
        loss_up  = self.loss_up.data[0]
        loss_down= self.loss_down.data[0]
        loss_color= self.loss_color.data[0]
        loss_type= self.loss_type.data[0]
        return OrderedDict([('loss_cls', loss_cls), ('loss_up', loss_up), ('loss_down', loss_down), ('loss_color', loss_color), ('loss_type', loss_type)])

    def isout(self, p):
        x = p[0]
        y = p[1]
        if not (0 <= x and x < self.opt.fineW and 0 <= y and y < self.opt.fineH):
            return True
        return False

    def decode_lanes(self, cls, up, down, color, type, threshold):
        fea_step_y   = self.opt.fineH / self.opt.feaH
        fea_step_x   = self.opt.fineW / self.opt.feaW
        slice_step_y = self.opt.fineH / self.opt.slicing

        lanes = []
        for h in xrange(self.opt.feaH):
            stdy = h * fea_step_y
            for w in xrange(self.opt.feaW):
                stdx = w * fea_step_x + 0.5 * fea_step_x
                prob = cls[0][h][w]
                co   = color[0][h][w]
                ty   = type[0][h][w]
                if prob < threshold:
                    continue
                lane = []

                # up lane
                up_len  = int(up[0][h][w]) + 1
                y_id    = 1
                y_start = stdy
                while y_id < self.opt.slicing + 1:
                    pt = [stdx + up[y_id][h][w], y_start]
                    lane.append(pt)
                    y_id    += 1
                    y_start -= slice_step_y

                    if y_id >= up_len: 
                        break
                    if self.isout(pt):
                        break


                lane.reverse()
                # down lane
                down_len = int(down[0][h][w]) + 1
                y_id     = 1
                y_start  = stdy + slice_step_y
                while y_id < self.opt.slicing + 1:
                    pt = [stdx + down[y_id][h][w], y_start]
                    lane.append(pt)
                    y_id    += 1
                    y_start += slice_step_y

                    if (y_id >= down_len): 
                        break
                    if self.isout(pt):
                        break

                lanes.append([lane, prob, co, ty])
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
        res  = []
        for i in xrange(le):
            if flag[i] == 1:
                continue

            la = las[i]
            res.append(la)
            for j in xrange(i + 1, le):
                if flag[j] == 1:
                    continue
                if self.dis(la[0], las[j][0]) < dis_thres:
                    flag[j] = 1
        return res

    def scale(self, lanes, oldH, oldW):
        y_ratio = 1.0 * oldH / self.opt.fineH
        x_ratio = 1.0 * oldW / self.opt.fineW
        reslanes = []
        for lane_group in lanes:
            reslane = []
            for p in lane_group[0]:
                reslane.append((p[0] * x_ratio, p[1] * y_ratio))
            #reslanes.append(reslane)
            obj = [reslane]
            for i in xrange(1, len(lane_group)):
                obj.append(lane_group[i])
            reslanes.append(obj)
        return reslanes

    #def draw_lanes(self, lanes):
    #    img = util.tensor2im(self.input.data[0])
    #    img = image_draw_line_list(img, lanes)
    #    return img
    def draw_lanes(self, lanes):
        img = util.tensor2im(self.input.data[0])
        img = image_draw_line_cls_list(img, lanes)
        return img

    def draw(self, cls, up, down, color, type, threshold):
        img = util.tensor2im(self.input.data[0])
        # TODO uncover the label into lanes
        lanes = self.decode_lanes(cls, up, down, color, type, threshold)
        lanes = self.nms(lanes, 10)
        img = image_draw_line_cls_list(img, lanes)
        return img

    def get_current_visuals(self, threshold):
        imgA = self.draw(self.pre_cls.data[0], self.pre_up.data[0], self.pre_down.data[0], self.pre_color.data[0], self.pre_type.data[0], threshold)
        imgB = self.draw(self.label_cls.data[0], self.label_up.data[0], self.label_down.data[0], self.label_color.data[0], self.label_type.data[0], threshold)
        return OrderedDict([('img_predict', imgA), ('img_label', imgB)])
        
    def save(self, label):
        self.save_network(self.net, 'resnext%d' % (self.opt.depth), label, self.gpu_ids)
