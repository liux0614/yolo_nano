import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

import  os

from options.train_options import TrainOptions
from data.get_dataset import get_dataset
from models.get_model import get_model


if __name__ == '__main__':
    opt = TrainOptions().parse() # get training options
    dataset = get_dataset(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )

    model = get_model(opt)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    num_batches = int(len(dataloader) / opt.batch_size)
    for epoch in range(opt.start_epochs, opt.start_epochs+opt.epochs):
        for i, (img_path, imgs, targets) in enumerate(dataloader):
            if len(opt.gpu_ids) > 0:
                model = model.to(opt.device)
                imgs = Variable(imgs.to(opt.device))
                if targets is not None:
                    targets = Variable(targets.to(opt.device), requires_grad=False)
            
            model.train()
            loss, yolo_outputs = model.forward(imgs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'epochs: {epoch}/{opt.epochs}, batch: {i}/{num_batches}')

            assert False