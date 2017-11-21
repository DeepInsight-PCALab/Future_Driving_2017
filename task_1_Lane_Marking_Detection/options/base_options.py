#!/usr/bin/env python
# coding=utf-8

import argparse
import os
from   util import util
import torch

class BaseOptions(object):
    def __init__(self):
        self.parser      = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

        #self.parser.add_argument('--dataroot', required = True, help = 'path to images: train/, val/, test/, train.json, val.json, test.json')
        self.parser.add_argument('--batchSize',   type = int, default = 10,  help = 'input batch size')
        self.parser.add_argument('--loadH',       type = int, default = 280, help = 'scale to H' )
        self.parser.add_argument('--loadW',       type = int, default = 350, help = 'scale to W' )
        self.parser.add_argument('--fineH',       type = int, default = 256, help = 'crop to H' )
        self.parser.add_argument('--fineW',       type = int, default = 320, help = 'crop to W' )
        self.parser.add_argument('--feaH',        type = int, default = 16,  help = 'feature for H')
        self.parser.add_argument('--feaW',        type = int, default = 20,  help = 'feature for W')
        self.parser.add_argument('--pos_thres',   type = int, default = 7,   help = 'positive threshold')
        self.parser.add_argument('--neg_thres',   type = int, default = 8,   help = 'negative threshold')
        self.parser.add_argument('--negpos_ratio',type = int, default = 10,  help = 'neg: pos ratio')
        self.parser.add_argument('--input_nc',    type = int, default = 3,   help = '# of input image channels')
        self.parser.add_argument('--slicing',     type = int, default = 64,  help = '# of slicing parallel lines')
        self.parser.add_argument('--model',       type = str, default = 'resnext_cls',      help = 'selects model to use')
        self.parser.add_argument('--depth',       type = int, default = 101, help = '# of resnext depth')
        self.parser.add_argument('--gpu_ids',     type = str, default = '0', help = 'gpu ids: 0; 0,1,2;')
        self.parser.add_argument('--name',        type = str, default = 'experiment_name', help = 'name of the exp')
        self.parser.add_argument('--nThreads',    type = int, default = 2,   help = '# threads for loading data')
        self.parser.add_argument('--checkpoints', type = str, default = './checkpoints',   help = 'models are saved here')
        self.parser.add_argument('--display_winsize', type = int, default = 256, help = 'display window size')
        self.parser.add_argument('--lam',             type = float, default = 0.1,     help = 'loss balance for cls loss')
        self.parser.add_argument('--lam_attr',        type = float, default = 0.2,     help = 'loss balance for cls loss')

    #self.parser.add_argument('--display_freq',     type = int, default = 10, help = 'frequency of showing training results on screen')
    #self.parser.add_argument('--print_freq',       type = int, default = 10, help = 'frequency of showing training results on console')
    #self.parser.add_argument('--save_latest_freq', type = int, default = 500,help = 'frequency of saving the latest results')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        if self.isTrain:
            self.opt.display_freq = self.opt.display_freq * self.opt.batchSize
            self.opt.print_freq   = self.opt.print_freq * self.opt.batchSize
            self.opt.save_latest_freq = self.opt.save_latest_freq * self.opt.batchSize
            # schedule assign
            str_schedules = self.opt.schedule.split(',')
            self.opt.schedules = []
            for str_schedule in str_schedules:
                s = int(str_schedule)
                self.opt.schedules.append(s)
            self.opt.schedule_max = self.opt.schedules[-1]

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        #if len(self.opt.gpu_ids) > 0:
        #    torch.cuda.set_device(self.opt.gpu_ids[0])
        args = vars(self.opt)
        print '------------------- Option ---------------------'
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print '------------------- End ------------------------'

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints, self.opt.name)
        util.mkdirs(expr_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------------- Option ---------------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('------------------- End ------------------------\n')

        return self.opt
