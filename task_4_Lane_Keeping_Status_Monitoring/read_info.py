# encoding=utf-8

import numpy as np
import os
import cv2
import sys
import json
# import xml.etree.cElementTree as ET
from lxml import etree as ET
import math

def get_data(st):
    pos_b = st.find("<data>")
    pos_e = st.find("</data>")
    tmp = ""
    for i in range(pos_b+6, pos_e):
        tmp = tmp + st[i]
    return tmp

def read_xml(info_path):
    xml_path = info_path
    print(xml_path)
    fin = open(xml_path, 'r')
    str = ""
    for line in fin:
        line = line.strip('\n').strip('\r')
        str = str + line
    fin.close()

    print(str)
    # camera parameters ------------------------------------------------
    pos_b = str.find("<camera_matrix")
    pos_e = str.find("</camera_matrix>")
    tmp = ""
    for i in range(pos_b+1, pos_e):
        tmp = tmp + str[i]
    param = get_data(tmp)
    param = param.split(' ')
    tmp_data = []
    for val in param:
        if val == "": continue
        tmp_data.append(float(val))
    print(tmp_data)
    camera_param = np.reshape(np.array(tmp_data), (3, 3))
    #print(camera_param)
            
    # rotation paramteres -------------------------------------------------
    pos_b = str.find("<rotation_matrix")
    pos_e = str.find("</rotation_matrix>")
    tmp = ""
    for i in range(pos_b+1, pos_e):
        tmp = tmp + str[i]
    param = get_data(tmp)
    param = param.split(' ')
    tmp_data = []
    for val in param:
        if val == "": continue
        tmp_data.append(float(val))
    rotation_param = np.reshape(np.array(tmp_data), (3, 3))
    # print(rotation_param)
        
    # translation parameters --------------------------------------------
    pos_b = str.find("<translation_vector")
    pos_e = str.find("</translation_vector>")
    tmp = ""
    for i in range(pos_b, pos_e):
        tmp = tmp + str[i]
    param = get_data(tmp)
    param = param.split(' ')
    tmp_data = []
    for val in param:
        if val == "": continue
        tmp_data.append(float(val))
    translation_param = np.reshape(np.array(tmp_data), (3, 1))
    # print(translation_param)
    
    return camera_param, rotation_param, translation_param


if __name__ == '__main__':
    read_xml()
 
