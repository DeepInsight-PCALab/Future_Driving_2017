#!/usr/bin/env python
# coding=utf-8

import os.path
#from data.utils       import make_dataset, is_image_file   # finish
from data.utils       import make_label     # finish
from data.transfer    import Transformer    # finish
from PIL import Image
import PIL
import random

import torch.utils.data as data

class LaneDataset(data.Dataset):
    def __init__(self, opt, phase): # 'phase = train/val/test'
        super(LaneDataset, self).__init__()

        self.opt   = opt
        self.phase = phase
        self.root  = opt.dataroot
        #self.dir   = os.path.join(opt.dataroot, phase)
        self.label_file = os.path.join(opt.dataroot, opt.json + '.json')
        print 'load label_file ...', self.label_file

        #self.paths  = make_dataset(self.dir)
        #self.paths  = sorted(self.paths)
        self.labels = make_label(self.label_file, opt.dataroot)
        self.paths  = []
        
        keys = self.labels.keys()
        deletes = []
        for image_path in keys:
            if os.path.exists(image_path):
                self.paths.append(image_path)
            else:
                deletes.append(image_path)

        for image_path in deletes:
            print('delete ', image_path)
            del self.labels[image_path]

        self.size        = len(self.paths)
        print('----------> total %s dataset number = ' % phase, self.size)
        self.transformer = Transformer(opt)

    def __getitem__(self, index):
        idx  = index % self.size
        path = self.paths[idx]
        img  = Image.open(path).convert('RGB')
        label = self.labels[path]

        img, label = self.transformer.gao(img, label, self.phase)  # we need to transfer label into an feature map
        #print('img = ', img, 'label = ', label)
        #return {"input": img, "label": label} #img, label
        d = {"input": img}
        d.update(label)
        return d

    def __len__(self):
        return self.size

    def name(self):
        return 'LaneDataset'
