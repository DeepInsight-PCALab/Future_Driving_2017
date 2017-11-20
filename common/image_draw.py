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


def image_draw_line_cls_list(image, line_cls_list, color = (0, 0, 255), thin = 2):
    img = image.copy()
    for line_cls in line_cls_list:
        line = line_cls[0]
        le = len(line)
        prob, co, ty = line_cls[1], line_cls[2], line_cls[3]
        if co > 0.5:
            color = (255, 255, 60)
        else:
            color = (0, 0, 255)

        if ty > 0.5: 
            for i in xrange(le - 1):
                pa = (int(line[i][0]), int(line[i][1]))
                pb = (int(line[i + 1][0]), int(line[i + 1][1]))
                cv2.line(img, pa, pb, color, thin)
        else:
            for i in xrange(0, le - 2, 4):
                pa = (int(line[i][0]), int(line[i][1]))
                pb = (int(line[i + 1][0]), int(line[i + 1][1]))
                pc = (int(line[i + 2][0]), int(line[i + 2][1]))
                cv2.line(img, pa, pb, color, thin)
                cv2.line(img, pb, pc, color, thin)
        most_left = line[0] if line[0][0] < line[-1][0] else line[-1]
        org = (int(line[le/2][0]), int(line[le/2][1]))
        org = (int((org[0] + most_left[0]) / 2.0), int((org[1] + most_left[1]) / 2.0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'P:%02d|Y:%02d|S:%02d' % (int(prob * 100), int(co * 100), int(ty * 100)), org, font, 0.4, (255, 0, 0), 2)

    res = 0.5 * image + 0.5 * img
    res = res.astype(np.uint8)
    return res


#img = image_draw_line_cls_list(img, lanes)
