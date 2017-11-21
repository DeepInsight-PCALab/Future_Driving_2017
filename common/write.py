#!/usr/bin/env python
# coding=gbk
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
class XmlWriter(object):
    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.doc = Document()
        self.root = self.doc.createElement('opencv_storage')
        self.doc.appendChild(self.root)
    
    def add_lanes(self, lanes_group, fid):
        number = len(lanes_group)
        add_son(self.doc, self.root, 'Frame%05dTargetNumber' % fid, str(number))
        for i in xrange(number):
            lane = lanes_group[i][0]
            prob, color, type = lanes_group[i][1], lanes_group[i][2], lanes_group[i][3]
            cnt  = len(lane)
            pt_array = ' '.join([str(x[0])+' '+str(x[1]) for x in lane])
            a = add_son_frame(self.doc, self.root, 'Frame%05dTarget%05d' % (fid, i))
            if color > 0.5:
                if type > 0.5:
                    s = u'\'黄色实线\''
                else:
                    s = u'\'黄色虚线\''
            else:
                if type > 0.5:
                    s = u'\'白色实线\''
                else:
                    s = u'\'白色虚线\''
            #s = u'\'白色实线\''
            add_son(self.doc, a, 'Type', s)
            b = add_son_frame_attr(self.doc, a, 'Position', 'type_id', "opencv-matrix")
            add_son(self.doc, b, 'rows', str(cnt))
            add_son(self.doc, b, 'cols', '1')
            add_son(self.doc, b, 'dt', '\'2i\'')
            add_son(self.doc, b, 'data', pt_array)

    def write_xml(self):
        f = codecs.open(self.xml_file + '.tmp', 'w', 'gbk')
        self.doc.writexml(f, addindent = '\t', newl = '\n', encoding = 'gbk')
        fr = codecs.open(self.xml_file + '.tmp', 'r', 'gbk')
        fw = codecs.open(self.xml_file, 'w', 'gbk')
        for line in fr:
            fw.write(line.replace('\'', '\"'))
        fr.close()
        fw.close()


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
        s = u'白色实线'
        #s = unicode(s, 'utf-8').decode('gbk') #.encode('gbk')
        #s = s.decode('utf-8').encode('gbk')
        #s = 'hehe'
        add_son(doc, a, 'Type', s)
        b = add_son_frame_attr(doc, a, 'Position', 'type_id', "opencv-matrix")
        add_son(doc, b, 'rows', str(cnt))
        add_son(doc, b, 'cols', '1')
        add_son(doc, b, 'dt', "\'2i\'")
        add_son(doc, b, 'data', pt_array)

    f = codecs.open('tmp.xml', 'w', 'gbk')
    doc.writexml(f, addindent = ' ', newl = '\n', encoding = 'gbk')
    #with codecs.open('tmp.xml', 'w', 'utf-8') as f:
    #    f.write(doc.toprettyxml(indent = '\t', encoding = 'utf-8'))
    #with codecs.open('tmp.xml', 'w', 'gbk') as f:
    #    f.write(doc.toprettyxml(indent = '\t', encoding = 'gbk'))

#writeInfoToXml([[1,2,3,4], [1,1,1,1,1,1,1,1]], 1)

