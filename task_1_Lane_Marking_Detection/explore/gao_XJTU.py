#!/usr/bin/env python
# coding=utf-8

import os, sys
sys.path.append('..')
sys.path.append('../..')
from common.utils import is_image_file

paths = [
    '/data3/XJTU2017/task_1/TSD-Lane/',
    '/data3/XJTU2017/task_2/TSD-Signal/',
    '/data3/XJTU2017/task_3/TSD-Vehicle/',
    '/data3/XJTU2017/task_4/TSD-LKSM/',
    '/data3/XJTU2017/task_5/TSD-FVDM/'
]

to = '/data3/DeepInsight/Lane_PAMI/Datasets/tolabel/'
import shutil

cnt = 0

for i, path in enumerate(paths):
    to_folder = os.path.join(to, 'T%02d' % (i + 1))
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)

    for root, _, files in os.walk(path):
        for file in files:
            #print(file)
            from_path = os.path.join(root, file)

            #to_name = file.replace('/', '|')
            to_path = os.path.join(to_folder, file)

            if is_image_file(file):
                if cnt % 100 == 0:
                    print(cnt)
                    print('from = ', from_path, 'to = ', to_path)
                cnt += 1
                #shutil.copy(from_path, to_path)

    print('now number = ', cnt)

print('total number = ', cnt)
                



