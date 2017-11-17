#!/usr/bin/env python
# coding=zh_CN.utf-8

import cv2
import numpy as np

from xml.dom.minidom import Document
import codecs

def add_son(doc,father, name, txt):
    dt = doc.createTextNode(txt)
    dn = doc.createElement(name)
    dn.appendChild(dt)
    father.appendChild(dn)
    return dn

def add_son_frame(doc,father, name):
    dn = doc.createElement(name)
    father.appendChild(dn)
    return dn

def add_son_frame_attr(doc, father, name, attr_name, attr_value):
    dn = doc.createElement(name)
    dn.setAttribute(attr_name, attr_value)
    father.appendChild(dn)
    return dn

#class 

def writeInfoToXml(lanes, fid):
    doc = Document()

    orderlist = doc.createElement('opencv_storage')
    doc.appendChild(orderlist)
    
    number = len(lanes)
    add_son(doc, orderlist, 'Frame%05dTargetNumber' % fid, str(number))
    for i in xrange(number):
        lane = lanes[i]
        cnt  = len(lane) / 2
        pt_array = ' '.join([str(x) for x in lane])
        a = add_son_frame(doc, orderlist, 'Frame%05dTarget%05d' % (fid, i))
        add_son(doc, a, 'Type', 'yellow-solid')
        b = add_son_frame_attr(doc, a, 'Position', 'type_id', "opencv-matrix")
        add_son(doc, b, 'rows', str(cnt))
        add_son(doc, b, 'cols', '1')
        add_son(doc, b, 'dt', "\'2i\'")
        add_son(doc, b, 'data', pt_array)

    #f = codecs.open('tmp.xml', 'w', 'zh_CN.UTF-8')
    #dom.writexml(f, addindent = ' ', newl = '\n', encoding = 'zh_CN.UTF-8')
    #with codecs.open('tmp.xml', 'w', 'utf-8') as f:
    #    f.write(doc.toprettyxml(indent = '\t', encoding = 'utf-8'))
    with codecs.open('tmp.xml', 'w', 'zh_CN.UTF-8') as f:
        f.write(doc.toprettyxml(indent = '\t', encoding = 'zh_CN.UTF-8'))


writeInfoToXml([[1,2,3,4], [1,1,1,1,1,1,1,1]], 1)

