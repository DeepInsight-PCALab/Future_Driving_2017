#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
def image_draw_line_list(image, line_list, color = (0, 0, 255), thin = 2):
    img = image.copy()
    for line in line_list:
        le = len(line)
        for i in xrange(le - 1):
            pa = (int(line[i][0]), int(line[i][1]))
            pb = (int(line[i + 1][0]), int(line[i + 1][1]))
            cv2.line(img, pa, pb, color, thin)
    res = 0.5 * image + 0.5 * img
    res = res.astype(np.uint8)
    return res

def image_draw_dot_line_list(image, line_list, color = (0, 0, 255), thin = 2):
    img = image.copy()
    for line in line_list:
        le = len(line)
        for i in xrange(0, le - 2, 4):
            pa = (int(line[i][0]), int(line[i][1]))
            pb = (int(line[i + 1][0]), int(line[i + 1][1]))
            pc = (int(line[i + 2][0]), int(line[i + 2][1]))
            cv2.line(img, pa, pb, color, thin)
            cv2.line(img, pb, pc, color, thin)

    res = 0.5 * image + 0.5 * img
    res = res.astype(np.uint8)
    return res

def image_draw_half_line_list(image, line_list, color = (0, 0, 255), thin = 2):
    img = image.copy()
    for line in line_list:
        le = len(line)
        for i in xrange((le - 1) / 2):
            pa = (int(line[i][0]), int(line[i][1]))
            pb = (int(line[i + 1][0]), int(line[i + 1][1]))
            cv2.line(img, pa, pb, color, thin)

    res = 0.5 * image + 0.5 * img
    res = res.astype(np.uint8)
    return res
