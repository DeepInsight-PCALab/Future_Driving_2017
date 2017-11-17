#!/usr/bin/env python
# coding=utf-8
from .resnext_model import ResNeXtModel

def create_model(opt):
    model = None
    print(opt.model)

    if opt.model == 'resnext':
        model = ResNeXtModel(opt)

    print 'model [%s] was created' % model.name()
    return model
