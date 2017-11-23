# encoding=utf-8

import numpy as np
import os
import cv2
import sys
import json
# import xml.etree.cElementTree as ET
from lxml import etree as ET
import math

from img2lanes import getLanes

def load_lane_image(path):
    img = cv2.imread(path)
    """
    CV_INTER_NN - 最近邻插值  
    CV_INTER_LINEAR - 双线性插值 (缺省使用)  
    CV_INTER_AREA - 使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN 方法..  
    CV_INTER_CUBIC - 立方插值
    """
    oh, ow, ch = img.shape
    oimg = img.copy()
    # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return img, ow, oh, oimg

def image_draw_line_list(image, y, x_line_list, left_lane=-1, right_lane=-1, new_u=-1, new_v=-1, color = (0, 0, 255), thin = 2):
    # print(image.shape)
    img = image.copy()
    # img = image
    idx = 0
    for line in x_line_list:
        end = len(line)
        start = 0
        for val in line:
            if val == -2:
                start += 1
            else:
                break
        end_bk = end
        for i in range(end-1, 0, -1):
            if line[i] == -2:
                end_bk -= 1
            else:
                break
        end = end_bk

        if idx == left_lane or idx == right_lane:
            color = (0, 0, 255)  # bgr  mian line
        else:
            color = (255, 0, 0)  # bgr  other line
        idx += 1
        for i in xrange(start, end - 1):
            pa = (int(line[i]), int(y[i]))
            pb = (int(line[i + 1]), int(y[i + 1]))
            cv2.line(img, pa, pb, color, thin)
    res = 0.5 * image + 0.5 * img

    # 画红色的填充圆,圆点（new_u, new_v）半径3
    res = cv2.circle(res, (new_u, new_v), 5, (0, 255, 0), -1)

    res = res.astype(np.uint8)
    return res

def detect_main_lane(y, x_line_list):
    k_set = np.zeros((len(x_line_list),))
    vis = np.zeros((len(x_line_list),), dtype=np.int)
    # print(len(x_line_list))
    i = 0
    for line in x_line_list:
        end = len(line)
        start = 0
        for val in line:
            if val == -2:
                start += 1
            else:
                break
        end_bk = end
        for j in range(end-1, 0, -1):
            if line[j] == -2:
                end_bk -= 1
            else:
                break
        end = end_bk
        ss = end - 2
        if start > ss:
            return (-1, -1)
        k, num = 0., 0
        # print(len(line), ss, end)
        # print(line)
        # print(y)
        for j in range(ss, end-1):
            # print(line[j], line[j+1])
            dy = float(line[j]) - float(line[j+1])
            # print(dy)
            dx = float(y[j]) - float(y[j+1])
            if dx >= -0.000001 and dx <= 0.000001: continue
            k += dy/dx
            num += 1
            # vis[j] = 1
        if num != 0:
            # print(k, num)
            k /= num
            k_set[i] = k
            vis[i] = 1
        else:
            vis[i] = 0
        i += 1

    clear_k_set = []
    clear_k_set_idx = []
    for i in range(k_set.shape[0]):
        if vis[i] == 0: continue
        clear_k_set.append(k_set[i])
        clear_k_set_idx.append(i)
    # print(clear_k_set)
    # print(clear_k_set_idx)
    clear_k_set = np.array(clear_k_set)
    clear_k_set_idx = np.array(clear_k_set_idx)
    left_lane, right_lane = -1, -1
    for i in range(clear_k_set.shape[0]-1):
        # print(clear_k_set[i]), 
        if clear_k_set[i] * clear_k_set[i+1] < 0:
            # print(clear_k_set[i], clear_k_set[i+1])
            if clear_k_set[i] < 0:
                left_lane = clear_k_set_idx[i]
                right_lane = clear_k_set_idx[i+1]
            else:
                right_lane = clear_k_set_idx[i]
                left_lane = clear_k_set_idx[i+1]
            break
    return (left_lane, right_lane)
    
def world2camera(x0, y0):
    u, v = 0, 0
    A = np.mat(np.array([[1652.7548306478222, 0., 645.59843662100081], [0., 1651.4166835094670, 476.44495746759020], [0., 0., 1.]]))
    Rt = np.array([[0.0079177846640020241, -0.99986396117327780, -0.014469548468994465, 3.6554495974876362],
                  [-0.11471346552052106, 0.013466264181992165, -0.99330734445953484, 223.23405430707408],
                  [0.99336706685609766, 0.0095246457079977854, -0.11459123705524082, -106.12491050729992]])
    R = np.mat(np.array([[0.0079177846640020241, -0.99986396117327780, -0.014469548468994465],
                  [-0.11471346552052106, 0.013466264181992165, -0.99330734445953484],
                  [0.99336706685609766, 0.0095246457079977854, -0.11459123705524082]]))
    t = np.mat(np.array([[3.6554495974876362], [223.23405430707408], [-106.12491050729992]]))
    Rt = np.mat(Rt)
    # M_ = np.mat(np.array([[1.0 * x0], [1.0 * y0], [106.]]))
    M_ = np.mat(np.array([[1.0 * x0], [1.0 * y0], [0.]]))
    fx = 1652.7548306478222
    fy = 1651.4166835094670
    cx = 645.59843662100081
    cy = 476.44495746759020
    xx = R * M_ + t
    x, y, z = xx[0], xx[1], xx[2]
    xp, yp = x/z, y/z
    u = int(fx * xp + cx)
    v = int(fy * yp + cy)
    if u >= 0 and u < 1280 and v >= 0 and v < 1024:
        return True, u, v
    else:
        return False, u, v

def expand_line_list(x_line_list):
    # 补齐线段后面的-2
    line_num = len(x_line_list)
    for i in range(line_num):
        len_line = len(x_line_list[i])
        k = -1
        for j in range(len_line-1, 0, -1):
            if x_line_list[i][j] != -2:
                k = j+1
                break
        # print(k)
        for j in range(k, len_line):
            x_line_list[i][j] = 2.*x_line_list[i][j-1] - x_line_list[i][j-2]
    return x_line_list

"""
def expand_y(y):
    # 补齐线段后面的-2
    line_num = len(y)
    for i in range(line_num):
        len_line = len(x_line_list[i])
        k = -1
        for j in range(len_line-1, 0, -1):
            if x_line_list[i][j] != -2:
                k = j+1
                break
        # print(k)
        for j in range(k, len_line):
            x_line_list[i][j] = 2.*x_line_list[i][j-1] - x_line_list[i][j-2]
    return x_line_list
"""

def lane_bubble_sort(x_line_list, up=True):
    lane_num = len(x_line_list)
    for i in range(0, lane_num):
        for j in range(0, lane_num-1):
            len_1, len_2 = len(x_line_list[j]), len(x_line_list[j+1])
            if up == True:
                if x_line_list[j][len_1-1] > x_line_list[j+1][len_2-1]:
                    tmp = x_line_list[j]
                    x_line_list[j] = x_line_list[j+1]
                    x_line_list[j+1] = tmp
            else:
                if x_line_list[j][len_1-1] < x_line_list[j+1][len_2-1]:
                    tmp = x_line_list[j]
                    x_line_list[j] = x_line_list[j+1]
                    x_line_list[j+1] = tmp
    return x_line_list

def calc_world2camera(x0, y0, ow, oh):
    u, v = 0, 0
    A = np.mat(np.array([[1652.7548306478222, 0., 645.59843662100081], [0., 1651.4166835094670, 476.44495746759020], [0., 0., 1.]]))
    Rt = np.array([[0.0079177846640020241, -0.99986396117327780, -0.014469548468994465, 3.6554495974876362],
                  [-0.11471346552052106, 0.013466264181992165, -0.99330734445953484, 223.23405430707408],
                  [0.99336706685609766, 0.0095246457079977854, -0.11459123705524082, -106.12491050729992]])
    R = np.mat(np.array([[0.0079177846640020241, -0.99986396117327780, -0.014469548468994465],
                  [-0.11471346552052106, 0.013466264181992165, -0.99330734445953484],
                  [0.99336706685609766, 0.0095246457079977854, -0.11459123705524082]]))
    t = np.mat(np.array([[3.6554495974876362], [223.23405430707408], [-106.12491050729992]]))
    Rt = np.mat(Rt)
    # M_ = np.mat(np.array([[1.0 * x0], [1.0 * y0], [106.]]))  # 以相机作为世界坐标系的中心
    M_ = np.mat(np.array([[1.0 * x0], [1.0 * y0], [0.]]))  # 以相机下方的地面作为世界坐标系的中心
    fx = 1652.7548306478222
    fy = 1651.4166835094670
    cx = 645.59843662100081
    cy = 476.44495746759020
    xx = R * M_ + t
    x, y, z = xx[0], xx[1], xx[2]
    xp, yp = x/z, y/z
    u = int(fx * xp + cx)
    v = int(fy * yp + cy)

    # if u >= 0 and u < ow and v >= 0 and v < oh:
    if u >= -304 and u < 2922 and v >= -304 and v < 2999:
        return True, u, v
    else:
        return False, u, v

def binary_search(bx, y, x_line_list, lane_num, ow, oh):
    far = bx
    L, R = -1, -1
    for i in range(-5000, 5000):
        flag, c_x, c_y = calc_world2camera(far, i, ow, oh)
        if flag == True:
            L = i
            break
    for i in range(5000, -5000, -1):
        flag, c_x, c_y = calc_world2camera(far, i, ow, oh)
        if flag == True:
            R = i
            break
    
    eps = 1
    L, R, mid = 1. * L, 1. * R, -1. * 1e9
    c_x, c_y = -1. * 1e9, -1. * 1e9
    # print('binary_search L: {}  R: {}\n'.format(L, R))
    base = 255
    while np.abs(R-L) > eps:
        mid = (L + R) / 2.
        flag, c_x, c_y = calc_world2camera(far, int(mid), ow, oh)
        # print('!', mid, c_x, c_y)
        if flag == False:
            print('ERROR!')
            break

        tmp_i = -1
        for i in range(len(y)-1):
            if int(c_y) >= int(y[i]) and int(c_y) <= int(y[i+1]):
                tmp_i = i
                break
        # print('c_y: {} tmp_i: {}  lane_num: {}\n'.format(c_y, tmp_i, lane_num))
        p_x, p_y = c_x, c_y
        a_x, a_y = x_line_list[lane_num][tmp_i], y[tmp_i]
        # print(a_x, a_y)
        # print(tmp_i)
        # print('len x ', len(x_line_list[lane_num]))
        # print('len y ', len(y))
        b_x, b_y = x_line_list[lane_num][tmp_i+1], y[tmp_i+1]
        # print(b_x, b_y)
        cross = (b_x - a_x)*(p_y - a_y) - (b_y - a_y)*(p_x - a_x)
        # print(cross)
        # print('-'*20)
        if cross >= 0.00000:  # point in line right
            R = mid
        # elif tmp_i == -1 or  cross <= 0:  # point in line left
        elif cross <= 0:  # point in line left
            L = mid

    # print('-'*30)
    # print(int(c_x), int(c_y))
    return [c_x], [c_y], L

def generate(se, lane_list, ow, oh):
    tmp = []
    for val in se:
        tmp.append(val)
    # tmp.append(oh)
    lens = len(tmp)
    for i in range(lens):
        for j in range(lens-1):
            if tmp[j] > tmp[j+1]:
                t = tmp[j]
                tmp[j] = tmp[j+1]
                tmp[j+1] = t
    # print(tmp)
    if tmp[lens-1] < oh:
        tmp.append(oh)
        lens += 1

    lane_num = len(lane_list)

    x_list = []
    for line in lane_list:
        tt = [-2 for val in range(lens)]
        # print(tt)
        for val in line:
            yy = int(val[1])
            for i in range(lens):
                if yy == tmp[i]:
                    tt[i] = val[0]
                    break
        # print(tt)
        # print('\n')
        x_list.append(tt)

    return tmp, x_list

if __name__ == '__main__':
    LANE_DETECT = getLanes()
    # x_l, x_h = -4000, 4000
    # y_l, y_h = -2000, 2000
    debug = False

    image_width, image_height = 1280, 1024
    _, u, v = calc_world2camera(600., 0, image_width, image_height)

    img_base_path = "/data3/XJTU2017/task_4/TSD-LKSM/"
    file = open('./pami.json', 'r')
    number = 0
    for line in file:
        line = line.strip('\n')
        data = json.loads(line)
        task = data["raw_file"].split('/')[0]
        if task == "T04":
            raw_file = data["raw_file"]
            tmp = raw_file.split('/')[1].split('-')[:-1]
            img_path = img_base_path + tmp[0] + '-' + tmp[1] + '-' + tmp[2] + '/' + raw_file.split('/')[1]
            print('image path: {}'.format(img_path))
            
            lane_list = LANE_DETECT.get_lanes(img_path)
            # print(lane_list)
            lane = []
            se = set()
            for line in lane_list:
                tmp = []
                for val in line:
                    tmp.append((val[0], val[1]))
                    if int(val[1]) not in se:
                        se.add(int(val[1]))
            img, ow, oh, oimg = load_lane_image(img_path)
            tmp_y, tmp_x_line_list = generate(se, lane_list, ow, oh)
            x_line_list = tmp_x_line_list  # data["lanes"]
            y = tmp_y  # np.array(data["h_samples"])
            # print('x', len(x_line_list[0]))
            # print('y', len(y))
            oimg = cv2.circle(oimg, (u, v), 3, (0, 255, 0), -1)  # bgr
            # cv2.imwrite("./oc.png", oimg)
            w, h = ow, oh

            fx = 1. * w / ow  # x 轴缩放系数
            fy = 1. * h / oh  # y 轴缩放系数
            if debug: print('fx: {}, fy: {}\n'.format(fx, fy))
            original_u, original_v = u, v
            new_u, new_v = int(fx * u), int(fy * v)
            if debug: print('new u: {} new v: {}\n'.format(new_u, new_v))

            original_x_line_list, original_y = [], []
            for val_y in y:
                original_y.append(1. * val_y / fy)
            for line in x_line_list:
                lens = len(line)
                tmp = []
                for val_x in line:
                    if val_x == -2:
                        tmp.append(-2)
                    else:
                        tmp.append(1. * val_x / fx)
                original_x_line_list.append(tmp)
            if debug: print('original_x_line_list: {}'.format(np.array(original_x_line_list).shape))
            
            original_x_line_list = expand_line_list(original_x_line_list)
            if debug: print('original_x_line_list: {}'.format(np.array(original_x_line_list).shape))
            
            if debug:
                minn, maxx = float(1e9), -float(1e9)
                for line in original_x_line_list:
                    # print(line)
                    for val in line:
                        minn = min(minn, val)
                        maxx = max(maxx, val)
                print('after expand: minn width: {} maxx width: {}\n'.format(minn, maxx))

            # 车道线排序
            original_x_line_list = lane_bubble_sort(original_x_line_list, up=True)
            # print(original_x_line_list)

            # 检测主车道线
            (left_lane, right_lane) = detect_main_lane(original_y, original_x_line_list)
            print('left lane index: {}  right lane index: {}'.format(left_lane, right_lane))

            
            front_car = 600
            # 二分查找左车道线
            l_c_x, l_c_y, left_lane_world_y = binary_search(front_car, original_y, original_x_line_list, left_lane, ow, oh)
            
            # 二分查找右车道线
            r_c_x, r_c_y, right_lane_world_y = binary_search(front_car, original_y, original_x_line_list, right_lane, ow, oh)
            distance = (left_lane_world_y+right_lane_world_y) / 2.
            print('distance: {}'.format(distance))
            
            oimg = image_draw_line_list(oimg, original_y, original_x_line_list, left_lane, right_lane, original_u, original_v)
            
            # for i in range(len(l_c_x)):
            #     oimg = cv2.circle(oimg, (l_c_x[i], l_c_y[i]), 5, (0, 0, 255), -1)  # 左侧车道线的焦点
            # for i in range(len(r_c_x)):
            #     oimg = cv2.circle(oimg, (r_c_x[i], r_c_y[i]), 5, (0, 0, 255), -1)  # 右侧车道线的焦点

            # 绘制左侧车道线的焦点
            oimg = cv2.circle(oimg, (l_c_x[0], l_c_y[0]), 5, (0, 0, 255), -1)
            # 绘制右侧车道线的焦点
            oimg = cv2.circle(oimg, (r_c_x[0], r_c_y[0]), 5, (0, 0, 255), -1)  
            
            # 绘制两侧车道的中点 绘制偏移距离
            ccx, ccy = int((l_c_x[0]+r_c_x[0])/2), int((l_c_y[0]+r_c_y[0])/2.)
            cv2.line(oimg, (ccx, ccy), (original_u, original_v), (255, 255, 0), 2)
            oimg = cv2.circle(oimg, (ccx, ccy), 5, (255, 122, 255), -1)
            oimg = cv2.circle(oimg, (original_u, original_v), 5, (0, 255, 0), -1)
            
            # 绘制距离文字
            cv2.putText(oimg, "D: %f cm" % (distance), (int((ccx+original_u)/2.)-90, int((ccy+original_v)/2.)-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (122, 255, 255), 2) #Draw the text
            cv2.putText(oimg, "Lane Center", (int(ccx)-160, int(ccy)-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 122, 255), 1) #Draw the text
            cv2.putText(oimg, "Base Point", (int(original_u)+100, int(original_v)-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1) #Draw the text
            # 保存图像
            cv2.imwrite("/data2/huile/pytorch/Future_Driving_2017/task_4_Lane_Keeping_Status_Monitoring/oc.png", oimg)
            print('{}\n'.format('*' * 100))

            number += 1
            if number == 30:
                break

    file.close()
 
