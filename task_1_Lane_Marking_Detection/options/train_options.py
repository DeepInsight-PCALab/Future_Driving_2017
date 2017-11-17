#!/usr/bin/env python
# coding=utf-8

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        #BaseOptions.__init__()

        self.parser.add_argument('--dataroot', required = True, help = 'path to images: train/, val/, test/, train.json, val.json, test.json')

        self.parser.add_argument('--display_freq',     type = int, default = 10, help = 'frequency of showing training results on screen')
        self.parser.add_argument('--print_freq',       type = int, default = 10, help = 'frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type = int, default = 500,help = 'frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq',  type = int, default = 1,   help = 'frequency of saving ckps at the end of epochs' )
        self.parser.add_argument('--continue_train',   action = 'store_true',     help = 'continue training, load the latest model')
        self.parser.add_argument('--epoch_count',      type = int, default = 1,   help = 'the starting epoch count, we save the model by <epoch_count>, <epoch_count> + <save_latest_freq>, ...')
        #self.parser.add_argument('--phase',            type = str, default = 'train', help = 'train, test')
        self.parser.add_argument('--which_epoch',      type = str, default = 'latest', help = 'which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--schedule',         type = str, default = '100,150,200', help = 'learning lr schedule')
        self.parser.add_argument('--schedule_max',     type = int, default = 200, help = 'learning lr schedule')
        self.parser.add_argument('--lr',               type = float, default = 0.01, help = 'initial learning rate for adam')
        self.parser.add_argument('--debug',            type = int, default = 0,     help = 'debug to print')
        #self.parser.add_argument('--no_html',          action = 'store_true',          help = 'do not save intermediate training results to [opt.checkpoints]/[opt.name]/web/')

        self.isTrain = True


        

