import os
import sys
import json
import numpy as np
import torch
import time
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from data.get_dataset import get_training_dataset, get_validation_set
from models.get_model import get_model
from transforms.transforms import Compose, MultiScale, ToTensor
from utils.opts import Opt
from utils.visualizer import Visualizer
from train import train

if __name__ == "__main__":

    opt = Opt().parse()
    ########################################
    #                 Model                #
    ########################################
    torch.manual_seed(opt.manual_seed)

    vis = Visualizer(opt)
    model = get_model(opt)
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    else:
        NotImplementedError("Only Adam and SGD are supported")
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=opt.lr_patience)
    
    ########################################
    #              Transforms              #
    ########################################
    transforms = Compose([
        MultiScale(opt.image_size), ToTensor()
    ])

    if opt.train:
        dataset = get_dataset(opt, transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_threads,
            collate_fn=dataset.collate_fn
        )

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.model == checkpoint['model']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    
    print('train model')
    for epoch in range(opt.begin_epoch, opt.num_epochs + 1):
        train(model, optimizer, dataloader, epoch, vis, opt)


        