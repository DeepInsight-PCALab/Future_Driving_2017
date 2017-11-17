#!/usr/bin/env python
# coding=utf-8
import os
import codecs
from bs4 import BeautifulSoup

def load_annotation(xml_filename):
    xml = ''
    with open(xml_filename) as f:
    #with codecs.open(xml_filename, 'r', encoding = 'utf-8') as f:
        xml = f.readlines()
    #print xml
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml)
