#!/usr/bin/env python
# coding=utf-8

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', 'JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
