import os
import time
import json
import torch
from torch import nn
from torch import optim

from data.get_dataset import get_train_dataset, get_val_dataset
from models.get_model import get_model
from transforms.get_transforms import get_train_transforms, get_val_transforms

from utils.opts import Opt
from utils.logger import Logger
from utils.visualizer import Visualizer

from train import train
from val import val

if __name__ == "__main__":

    opt = Opt().parse()

    ########################################
    #                 Model                #
    ########################################
    torch.manual_seed(opt.manual_seed)

    if opt.no_vis:
        visualizer = None
    else:
        visualizer = Visualizer(opt)
    model = get_model(opt)
    if opt.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr)
    else:
        NotImplementedError("Only Adam and SGD are supported")
    
    ########################################
    #              Transforms              #
    ########################################
    if not opt.no_train:
        train_transforms = get_train_transforms(opt)
        dataset = get_train_dataset(opt, train_transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_threads,
            collate_fn=dataset.collate_fn
        )
        train_logger = Logger(os.path.join(opt.checkpoint_path, 'train.log'))
            
    
    if not opt.no_val:
        val_transforms = get_val_transforms(opt)
        dataset = get_val_dataset(opt, val_transforms)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_threads,
            collate_fn=dataset.collate_fn
        )
        val_logger = Logger(os.path.join(opt.checkpoint_path, 'val.log'))

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.model == checkpoint['model']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])


    ########################################
    #           Train, Val, Test           #
    ########################################    
    for epoch in range(opt.begin_epoch, opt.num_epochs + 1):
        if not opt.no_train:
            print("\n---- Training Model ----")
            train(model, optimizer, dataloader, epoch, opt, train_logger, visualizer)

        if not opt.no_val:
            print("\n---- Evaluating Model ----")
            val(model, optimizer, dataloader, epoch, opt, val_logger, visualizer)

    
    if opt.test:
        pass