#!/usr/bin/env python
# coding=utf-8

import torch.utils.data as data
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', 'JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

#self.labels = make_label(self.label_file)
import json
import collections
def make_label(label_json, root):
    res = collections.OrderedDict()
    with open(label_json) as data_file:
        for line in data_file:
            anno = json.loads(line)
            raw = os.path.join(root, anno['raw_file'])
            res[raw] = anno
    return res
