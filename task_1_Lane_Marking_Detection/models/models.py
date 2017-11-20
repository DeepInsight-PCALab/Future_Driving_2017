#!/usr/bin/env python
# coding=utf-8
from .resnext_model import ResNeXtModel
from .resnext_cls_model import ResNeXtClsModel

def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'resnext':
        model = ResNeXtModel(opt)
    elif opt.model == 'resnext_cls':
        model = ResNeXtClsModel(opt)

    print 'model [%s] was created' % model.name()
    return model
