#!/usr/bin/env python
# coding=utf-8

import sys
sys.path.append('..')
sys.path.append('../..')
from common import load_annotation, image_draw_line_list
import os
import xml.dom.minidom
import cv2

img_root = '/data3/XJTU2017/task_1/TSD-Lane/TSD-Lane-00096/'
img_xml  = '/data3/XJTU2017/task_1/TSD-Lane-GT/TSD-Lane-00096-GT.xml'
output_root = './test_imgs/'
'''
import locale
print sys.getdefaultencoding()    #系统默认编码
print sys.getfilesystemencoding() #文件系统编码
print locale.getdefaultlocale()   #系统当前编码
print sys.stdin.encoding          #终端输入编码
print sys.stdout.encoding         #终端输出编码
import codecs
import io
f = codecs.open(img_xml, 'r', encoding = 'zh_CN.UTF-8')
#f = io.open(img_xml, 'r', encoding = 'zh_CN.UTF-8')
print f.readlines()
exit()
'''

def str2list(data):
    data = data.split(' ')
    ndata = []
    for v in data:
        if v != '':
            ndata.append(int(v))
    data = ndata
    return ndata

def generate_labeled_images(img_root, img_xml, output_root):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    anno = load_annotation(img_xml)
    #print(anno)
    sons = os.listdir(img_root)
    cnt = 0
    for son in sons:
        if cnt > 110:
            break
        cnt += 1

        img_path = os.path.join(img_root, son)
        img_to   = os.path.join(output_root, son)
        img_id   = img_path.split('-')[-1].split('.')[0]


        img      = cv2.imread(img_path)

        target   = 'frame%stargetnumber' % img_id
        num      = int(anno.findAll(target)[0].contents[0])
        #print num
        total_li = []
        total_ri = []
        for i in xrange(num):
            target_line = 'frame%starget%05d' % (img_id, i)
            line = anno.findAll(target_line)[0]
            
            data = line.findChildren('leftpoints')[0].findChildren('data')[0].contents[0].strip()
            data = str2list(data)
            #data = [int(v) if v!='' else  for v in data]
            li   = []
            for i in xrange(0, len(data), 2):
                li.append((data[i], data[i + 1]))
            total_li.append(li)

            data = line.findChildren('rightpoints')[0].findChildren('data')[0].contents[0].strip()
            data = str2list(data)
            ri   = []
            for i in xrange(0, len(data), 2):
                ri.append((data[i], data[i + 1]))
            total_ri.append(ri)


        new_img = image_draw_line_list(img, total_li, (0, 0, 255), 3)
        new_img = image_draw_line_list(new_img, total_ri, (255, 0, 0), 3)
        #new_img = cv2.resize(new_img, (1280, 720))
        cv2.imwrite(img_to, new_img)
        print('writing img ', img_to, 'with ', num, 'lines')
            #print(data)
            #print(type(data))
            #type = line.findChildren('type')[0].contents[0]
            #type = type.encode('zh_CN.UTF-8')
            #type = type.decode('gbk')
            #print(type)
            #print'白色实线'
            #if str(type) == str('白色实线'):
            #    print type
            #print line
'''
def generate_labeled_images(img_root, img_xml, output_root):
    domtree = xml.dom.minidom.parse(img_xml)
    anno    = domtree.documentElement

    sons = os.listdir(img_root)
    for son in sons:
        img_path = os.path.join(img_root, son)
        img_id   = img_path.split('-')[-1].split('.')[0]

        target   = 'Frame%sTargetNumber' % img_id
        num = anno.getElementsByTagName(target)
        print num
        num = num[0].data
        print num

'''
generate_labeled_images(img_root, img_xml, output_root)





