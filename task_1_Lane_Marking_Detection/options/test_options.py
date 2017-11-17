#!/usr/bin/env python
# coding=utf-8
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()

        self.parser.add_argument('--ntest', type = int, default = float('inf'), help = '# of test examples.')
        self.parser.add_argument('--test_dir', type = str, default = '/data3/XJTU2017/task_1/TSD-Lane/', help = 'test dir root')
        self.parser.add_argument('--results_dir', type = str, default = './results/', help = 'saves')
        self.parser.add_argument('--which_epoch', type = str, default = 'latest', help = 'which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--cls_thres', type = float, default = 0.5, help = 'lane prob at least cls_thres')

        self.isTrain = False
