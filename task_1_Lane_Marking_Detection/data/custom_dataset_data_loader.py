#!/usr/bin/env python
# coding=utf-8

import torch.utils.data
from data.lane_dataset import LaneDataset

class CustomDatasetDataLoader():
    def __init__(self, opt, phase):
        self.opt = opt
        self.dataset = LaneDataset(opt, phase)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size  = opt.batchSize,
            shuffle     = True,
            num_workers = int(opt.nThreads))

    def load_data(self):
        return self.dataloader
    
    def __len__(self):
        return len(self.dataset)


