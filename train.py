import  os
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

from data.get_dataset import get_dataset
from models.get_model import get_model
from utils.visualizer import Visualizer

def train(model, optimizer, dataloader, epoch, vis, opt):
    for i, (imgs, targets) in enumerate(dataloader):
            if len(opt.gpu_ids) > 0:
                imgs_cpu = imgs.clone()
                model = model.to(opt.device)
                imgs = Variable(imgs.to(opt.device))
                if targets is not None:
                    targets = Variable(targets.to(opt.device), requires_grad=False)
            
            loss, yolo_outputs = model.forward(imgs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            error52 = model.yolo_layer52.metrics
            error26 = model.yolo_layer26.metrics
            error13 = model.yolo_layer13.metrics
            vis.print_current_losses([error52, error26, error13], epoch, i, len(dataloader))
            vis.plot_current_visuals(imgs_cpu, yolo_outputs)

    # save checkpoints
    if epoch % opt.checkpoint_interval == 0:
        save_file_path = os.path.join(opt.checkpoint_path, 'epoch_{}.pth'.format(epoch+1))
        states = {
            'epoch': epoch + 1,
            'model': opt.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
            
