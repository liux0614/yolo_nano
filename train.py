import  os
import torch
import torch.nn as nn 
from torch.autograd import Variable

def train(model, optimizer, dataloader, epoch, vis, opt):
    for i, (images, targets) in enumerate(dataloader):
        # targets: [idx, class_id, x, y, h, w] in yolo format
        # idx is used to associate the bounding boxes with its image
        if not opt.no_cuda:
            images_cpu = images.clone()
            model = model.to(opt.device)
            images = Variable(images.to(opt.device))
            if targets is not None:
                targets = Variable(targets.to(opt.device), requires_grad=False)
        
        loss, yolo_outputs = model.forward(images, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        error52 = model.yolo_layer52.metrics
        error26 = model.yolo_layer26.metrics
        error13 = model.yolo_layer13.metrics
        vis.print_current_losses([error52, error26, error13], epoch, i, len(dataloader))
        vis.plot_current_visuals(images_cpu, yolo_outputs)

    # save checkpoints
    if epoch % opt.checkpoint_interval == 0:
        save_file_path = os.path.join(opt.checkpoint_path, 'epoch_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'model': opt.model,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
            
