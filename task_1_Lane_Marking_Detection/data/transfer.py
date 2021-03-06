#!/usr/bin/env python
# coding=utf-8
import numpy as np
from PIL import Image
import random
from collections import OrderedDict
import torchvision.transforms as transforms
import torch
import itertools

class Transformer():
    def __init__(self, opt):
        self.opt = opt
        self.opt.dh = self.opt.loadH - self.opt.fineH
        self.opt.dw = self.opt.loadW - self.opt.fineW
        assert self.opt.dh >= 0 and self.opt.dw >= 0, 'Warning: crop must be smaller or equal to load size!'
        self.randv  = []
        for i in xrange(self.opt.feaH):
            for j in xrange(self.opt.feaW):
                self.randv.append((i, j))

        transforms_list = []
        transforms_list += [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.totensor = transforms.Compose(transforms_list)

    def isout(self, p):
        x = p[0]
        y = p[1]
        if 0 <= x and x < self.opt.fineW + 1 and 0 <= y and y < self.opt.fineH + 1:
            return False
        return True

    def label_feature(self, lane_array, attr_array):
        eps = 1e-5
        # try to find positive samples
        record     = np.ones((self.opt.feaH, self.opt.feaW)) * (-1.0)
        record_min = np.ones((self.opt.feaH, self.opt.feaW)) * float('inf')

        color      = torch.zeros(self.opt.feaH, self.opt.feaW)
        color_mask = torch.zeros(self.opt.feaH, self.opt.feaW)
        type       = torch.zeros(self.opt.feaH, self.opt.feaW)
        type_mask  = torch.zeros(self.opt.feaH, self.opt.feaW)
        
        slice_step_y = self.opt.fineH / self.opt.slicing
        y_slices = range(0, self.opt.fineH, slice_step_y)       
        assert len(y_slices) == self.opt.slicing, 'Warning: slicing number wrong!'

        fea_step_y = self.opt.fineH / self.opt.feaH
        fea_step_x = self.opt.fineW / self.opt.feaW

        lane_dict = [] # project y->x, y in slices
        for lane_id, lane in enumerate(lane_array):
            lanexs, laneys = lane[0], lane[1]
            if len(laneys) == 0:
                print lane[0], lane[1]
            yid = 0
            ylength = len(laneys)
            ldict = OrderedDict()
            for slice in y_slices:
                while yid + 1 < ylength and laneys[yid + 1] < slice:
                    yid += 1
                if laneys[yid] < slice + eps and yid + 1 < ylength and laneys[yid + 1] > slice - eps:
                    # slice -> x
                    x = lanexs[yid + 1] + 1.0 * (lanexs[yid] - lanexs[yid + 1]) / (laneys[yid + 1] - laneys[yid]) * (laneys[yid + 1] - slice)
                    ldict[slice] = x
                    
                    # (x, slice) to check positive samples
                    fea_x = int(x)     / fea_step_x
                    fea_y = int(slice) / fea_step_y
                    # not every slice has to be an grid !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    if slice % fea_step_y == 0 and 0 <= fea_x and fea_x < self.opt.feaW and 0 <= fea_y and fea_y < self.opt.feaH:
                        # in fea grid
                        assert fea_step_x % 2 == 0
                        dis = abs(x - fea_x * fea_step_x - fea_step_x / 2)
                        if dis < record_min[fea_y][fea_x]: # and dis < self.opt.pos_thres:
                            record_min[fea_y][fea_x] = dis
                            record[fea_y][fea_x]     = lane_id
                            color[fea_y][fea_x]      = attr_array[lane_id][0]
                            type[fea_y][fea_x]       = attr_array[lane_id][1]

                            # check legal ''' tmp = color[fea_y][fea_x] assert tmp == 0 or tmp == 1, 'color = %d, not right' % (tmp) tmp = type[fea_y][fea_x] assert tmp == 0 or tmp == 1, 'type = %d, not right' % (tmp) '''
                            color_mask[fea_y][fea_x]  = 1.0
                            co = color[fea_y][fea_x]
                            if not (co == 0 or co == 1):
                                color_mask[fea_y][fea_x] = 0.0

                            type_mask[fea_y][fea_x]  = 1.0
                            ty = type[fea_y][fea_x]
                            if not (ty == 0 or ty == 1):
                                type_mask[fea_y][fea_x] = 0.0

            # append up 3 points
            keys = ldict.keys()
            keys.reverse()
            if len(keys) >= 2:
                for i in xrange(3):
                    k1 = keys[-1]
                    k2 = keys[-2]
                    nv = 2.0 * k1 - k2
                    if nv > 0:
                        ldict[nv] = 2.0 * ldict[k1] - ldict[k2]
                        keys.append(nv)
                    else:
                        break
            lane_dict.append(ldict)

        # filling class matrix
        # balance class
        pos_num = (record >= 0).sum()
        neg_mask= (record == -1.0)
        neg_num = neg_mask.sum()
        ignore_num = neg_num - pos_num * self.opt.negpos_ratio
        if ignore_num > 0:
            random.shuffle(self.randv)
            cnt = 0
            for i, v in enumerate(self.randv):
                y, x = v[0], v[1]
                if neg_mask[y][x]:
                    record[y][x] = -2.0
                    cnt += 1
                    if cnt == ignore_num:
                        break

        # now record is good, get it into classification label
        cls = np.copy(record)
        cls[record >= 0]  = 1  # positive
        cls[record == -1] = 0  # negative
        cls[record == -2] = -1 # ignore
        cls_mask = np.copy(cls)
        cls_mask[cls >= 0] = 1.0
        cls_mask[cls <  0] = 0.0

        res_cls      = torch.from_numpy(cls).unsqueeze(0).float() #.unsqueeze(0).float()
        res_cls_mask = torch.from_numpy(cls_mask).unsqueeze(0).float() #.unsqueeze(0).float()
        res_color    = color.unsqueeze(0).float()
        res_type     = type.unsqueeze(0).float()
        #res_attr_mask= attr_mask.unsqueeze(0).float()
        res_color_mask = color_mask.unsqueeze(0).float()
        res_type_mask  = type_mask.unsqueeze(0).float()

        # filling up, down
        res_up        = torch.zeros((self.opt.slicing + 1, self.opt.feaH, self.opt.feaW))
        res_up_mask   = torch.zeros((self.opt.slicing + 1, self.opt.feaH, self.opt.feaW))
        res_down      = torch.zeros((self.opt.slicing + 1, self.opt.feaH, self.opt.feaW))
        res_down_mask = torch.zeros((self.opt.slicing + 1, self.opt.feaH, self.opt.feaW))

        #slice_step_y = self.opt.fineH / self.opt.slicing
        fea_step_y    = self.opt.fineH / self.opt.feaH
        fea_step_x    = self.opt.fineW / self.opt.feaW
        for h in xrange(self.opt.feaH):
            stdy = h * fea_step_y
            for w in xrange(self.opt.feaW):
                stdx = w * fea_step_x + 0.5 * fea_step_x
                lane_id = int(record[h][w])
                if lane_id < 0:
                    continue
                ld   = lane_dict[lane_id]

                # for up
                y_start = stdy
                y_id    = 0
                scale   = 1.0
                while True:
                    if not y_start in ld.keys():
                        break
                    dx = ld[y_start] - stdx
                    res_up[y_id + 1][h][w]      = dx
                    res_up_mask[y_id + 1][h][w] = 1 * scale
                    scale   += 0.1
                    y_id    += 1
                    y_start -= slice_step_y
                # number to regress
                res_up[0][h][w]      = y_id
                res_up_mask[0][h][w] = 1
                
                y_start = stdy + slice_step_y
                y_id    = 0
                out_time = 0
                out_thres = 8
                while True:
                    if not y_start in ld.keys():
                        break
                    dx = ld[y_start] - stdx
                    res_down[y_id + 1][h][w]      = dx
                    res_down_mask[y_id + 1][h][w] = 1
                    y_id    += 1
                    # down add out_thres more out most
                    if self.isout([ld[y_start], y_start]):
                        out_time += 1
                    if out_time == out_thres:
                        break
                    y_start += slice_step_y

                # add more
                for i in xrange(out_thres - out_time):
                    if y_id - 1 > 0 and y_id + 1  < self.opt.slicing + 1:
                        res_down[y_id + 1][h][w]      = 2.0 * res_down[y_id][h][w] - res_down[y_id - 1][h][w]
                        res_down_mask[y_id + 1][h][w] = 1
                        y_id   += 1
                    else:
                        break

                # number to regress
                res_down[0][h][w]      = y_id
                res_down_mask[0][h][w] = 1

        d = {'cls': res_cls, 'cls_mask': res_cls_mask, 'up': res_up, 'up_mask': res_up_mask, 'down': res_down, 'down_mask': res_down_mask,
             'color': res_color, 'color_mask': res_color_mask, 'type': res_type, 'type_mask': res_type_mask}
        #return res_cls, res_cls_mask, res_up, res_up_mask, res_down, res_down_mask
        return d

    # last < 0 to be good value
    #    [-2, -2, 570, 554, 538, ..., 33, 16, -2,  -2, -2]
    # =>         [570, 554, 538, ..., 33, 16, -1, -18, -35]
    # =>         [y2,  y3,  y4,  ..., y...]
    def last_deal(self, xs, ys):
        le = xs.size
        pos_flag = False
        st = -1
        for i in xrange(le):
            if (not pos_flag) and xs[i] > 0:
                pos_flag = True
                st       = i
            if pos_flag and xs[i] < 0:
                #assert xs[i - 1] > 0 and xs[i - 2] > 0, 'Warning: before two elements are negative!'
                xs[i] = 2.0 * xs[i - 1] - xs[i - 2]
        #if st == -1 and self.opt.debug == 1:
        #    print xs
        #assert st != -1, 'st must be >= 0!'
        need_delete = False
        if st == -1:
            need_delete = True
            st = 0
        return np.array(xs[st: le]), np.array(ys[st: le]), need_delete

    # parse label into array, fitting, and calculate the last < 0
    def parse_label(self, label):
        y_mul = 1.0 / label['originH'] * self.opt.loadH
        x_mul = 1.0 / label['originW'] * self.opt.loadW
        lane_array = []
        attr_array = []
        std_ys = np.array(label['h_samples']) * y_mul

        assert (len(label['lanes']) == len(label['color']))
        assert (len(label['lanes']) == len(label['type']))

        for lane, co, ty in itertools.izip(label['lanes'], label['color'], label['type']):
            #print('lane x len = ', len(lane))
            #if self.opt.debug == 1:
            #    print np.array(lane)
            xs = np.array(lane) * x_mul
            # recalculate and short array, make last < 0 to be good value
            if (len(xs) != len(std_ys)):
                print(xs, std_ys)
            assert(len(xs) == len(std_ys))
            xs, ys, need_delete = self.last_deal(xs, std_ys)
            if not need_delete:
                lane_array.append([xs, ys])
                attr_array.append([co, ty])
        return lane_array, attr_array


    def gao_image_path(self, image_path):
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        self.original_width = width
        self.original_height = height
        if width != self.opt.fineW or height != self.opt.fineH:
            #image = image.resize((self.opt.loadW, self.opt.loadH), Image.BILINEAR)
            image = image.resize((self.opt.fineW, self.opt.fineH), Image.BILINEAR)
        image = self.totensor(image)
        return image


    def gao(self, image, label, phase):
        lane_array, attr_array = self.parse_label(label)
        if phase == 'train':
            width, height = image.size
            if width != self.opt.loadW or height != self.opt.loadH:
                image = image.resize((self.opt.loadW, self.opt.loadH), Image.BILINEAR)

            # ------------------------------------------------------------------------
            # crop !
            # parallel gao image and label
            dx = random.randint(0, self.opt.dw)
            dy = random.randint(0, self.opt.dh)
            image = image.crop((dx, dy, dx + self.opt.fineW, dy + self.opt.fineH))

            for i in xrange(len(lane_array)):
                lane_array[i][0] = lane_array[i][0] - dx
                lane_array[i][1] = lane_array[i][1] - dy

            # ------------------------------------------------------------------------
            # horizontal flip !
            # parallel gao image and label
            flip_flag = random.randint(0, 100)
            if flip_flag > 50:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                for i in xrange(len(lane_array)):
                    lane_array[i][0] = self.opt.fineW - 1 - lane_array[i][0]
        
        image = self.totensor(image)
        # ------------------------------------------------------------------------
        # cutout!
        if self.opt.cutout > 0:
            C, height, width = image.shape
            assert(C == 3 and height == self.opt.fineH and width == self.opt.fineW)
            cx = random.randint(0, width - 1)
            cy = random.randint(int(2.0 / 3 * height), height - 1)
            edge = self.opt.cutout / 2
            lp = (cx - edge if cx > edge else 0, cy - edge if cy > edge else 0)
            rp = (cx + edge if cx + edge <= width else width, cy + edge if cy + edge <= height else height)
            # cutout
            image[:, lp[1]: rp[1], lp[0]: rp[0]] = 0

        label = self.label_feature(lane_array, attr_array)
        #print('image.shape = ', image.shape)
        return image, label
