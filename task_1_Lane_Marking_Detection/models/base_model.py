#!/usr/bin/env python
# coding=utf-8

import os, torch

class BaseModel(object):
    def name(self):
        return 'BaseModel'

    def __init__(self, opt):
        self.opt     = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        if self.gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in self.gpu_ids])
            print('CUDA_VISIBLE_DEVICES = %s' % os.environ['CUDA_VISIBLE_DEVICES'])
        self.Tensor  = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir= os.path.join(opt.checkpoints, opt.name)

        #print('self.opt = ', self.opt)

    def set_input_and_label(self, dic):
        pass

    def forward(self):
        pass

    # for test
    # def test(self): pass

    # def get_image_paths(self):
    #    pass

    def optimize_parameters(self):
        pass

    def get_current_errors(self):
        return {}

    def get_current_visuals(self):
        pass

    def save(self, label):
        pass

    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path     = os.path.join(self.save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        #torch.save(network.cpu().state_dict(), save_path)
        #if len(gpu_ids) and torch.cuda.is_available():
        #    network.cuda(device_id = gpu_ids[0])

    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path     = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate(self, epoch):
        if epoch in self.opt.schedules:
            lr = self.optimizer.param_groups[0]['lr']
            print('learing rate %f --> %f' % ( lr, lr * 0.1 ))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
