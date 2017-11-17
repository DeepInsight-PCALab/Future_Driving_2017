#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os
import ntpath
import time
from . import util
from . import html

class Visualizer():
    def __init__(self, opt):
        self.use_html = opt.isTrain
        self.win_size = opt.display_winsize
        self.name     = opt.name
        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(opt.checkpoints, opt.name, 'loss_log.txt')
        with open(self.log_name, 'a') as log_file:
            now = time.strftime('%c')
            log_file.write('=================== Training Loss (%s) ===================\n' % now)


    def display_current_results(self, visuals, epoch):
        if self.use_html:
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                #print('image_numpy = ', image_numpy)
                #print('type()image_numpy = ', type(image_numpy))
                #print('image_numpy.shape = ', (image_numpy).shape)
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh = 1)

            for n in xrange(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []
                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width = self.win_size)
            webpage.save()

    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f)' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.5f|' % (k, v)

        print(message)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % message)






