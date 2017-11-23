#!/usr/bin/env python
# coding=utf-8
#from xml.dom.minidom import Document
import codecs

#class 
class XmlWriter(object):
    def __init__(self, xml_file):
        self.xml_file   = xml_file
        self.write_list = []

    def add_lanes(self, lanes_group, fid):
        self.write_list.append((lanes_group, fid))

    def write_xml(self):
        f = codecs.open(self.xml_file, 'w', 'gbk')
        f.write('<?xml version="1.0" encoding="gbk"?>\n')
        f.write('<opencv_storage>\n')
        frame_number = len(self.write_list)
        for j in xrange(frame_number):
            lanes_group = self.write_list[j][0]
            fid         = self.write_list[j][1]
            number      = len(lanes_group)
            f.write('<Frame%05dTargetNumber>%d</Frame%05dTargetNumber>\n'  % (fid, number, fid))
            for i in xrange(number):
                lane = lanes_group[i][0]
                prob, color, type = lanes_group[i][1], lanes_group[i][2], lanes_group[i][3]
                cnt  = len(lane)
                pt_array = ' '.join([str(x[0])+' '+str(x[1]) for x in lane])
                f.write('<Frame%05dTarget%05d>\n' % (fid, i))
                if color > 0.5:
                    if type > 0.5:
                        s = u'黄色实线' #.decode('utf-8').encode('gbk')
                    else:
                        s = u'黄色虚线'
                else:
                    if type > 0.5:
                        s = u'白色实线'
                    else:
                        s = u'白色虚线'
                #f.write('\t<Type>\"%s\"</Type>\n' % str(s).decode('utf-8').encode('gbk'))
                #print s.encode('gbk')
                #f.write('\t<Type>"%s"</Type>\n' % (s)) #.encode('gbk'))
                f.write('\t<Type>"%s"</Type>\n' % (s)) #.encode('gbk'))
                f.write('\t<Position type_id=\"opencv-matrix\">\n')
                f.write('\t\t<rows>%d</rows>\n' % cnt)
                f.write('\t\t<cols>1</cols>\n')
                f.write('\t\t<dt>"2i"</dt>\n')
                f.write('\t\t<data>%s</data>\n' % pt_array)
                f.write('\t</Position>\n')
                f.write('</Frame%05dTarget%05d>\n' % (fid, i))
        f.write('</opencv_storage>\n')
        f.close()

'''
tmp = XmlWriter('tmp.xml')
tmp.add_lanes([([(1,2),(3,4)], 0.9, 0.6, 0.3), ([(3,4),(6,7)], 0.9, 0.3, 0.8)], 0)
tmp.add_lanes([([(1,2),(3,4)], 0.9, 0.6, 0.3), ([(3,4),(6,7),(8,9)], 0.9, 0.3, 0.4)], 1)
tmp.write_xml()
exit()
class XmlWriterBak(object):
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
'''
