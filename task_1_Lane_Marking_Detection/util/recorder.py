#!/usr/bin/env python
# coding=utf-8

class Recorder():
    def __init__(self):
        self.D = {}
        self.cnt = 0

    def clear(self):
        self.D   = {}
        self.cnt = 0
    
    def add(self, errors):
        for k, v in errors.items():
            if not k in self.D:
                self.D[k] = 0.0

            self.D[k] += v

        self.cnt += 1

    def summary(self):
        res = ''
        for k in self.D.keys():
            res += '%s=%.3f' % (k, 1.0 * self.D[k] / self.cnt)
        return res




