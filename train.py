import  os
import torch
import torch.nn as nn 
from torch.autograd import Variable

from terminaltables import AsciiTable


def train(model, optimizer, dataloader, epoch, opt, logger, visualizer=None):
    for i, (images, targets) in enumerate(dataloader):
        batches_done = len(dataloader) * epoch + i
        # targets: [idx, class_id, x, y, h, w] in yolo format
        # idx is used to associate the bounding boxes with its image
        if not opt.no_cuda:
            images_cpu = images.clone()
            model = model.to(opt.device)
            images = Variable(images.to(opt.device))
            if targets is not None:
                targets = Variable(targets.to(opt.device), requires_grad=False)
        
        loss, yolo_outputs = model.forward(images, targets)
        loss.backward()

        if batches_done % opt.gradient_accumulations == 0:
            # accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()

        # logging
        metric_keys = model.yolo_layer52.metrics.keys()
        yolo_metrics = [model.yolo_layer52.metrics, model.yolo_layer26.metrics, model.yolo_layer13.metrics]
        metric_table_data = [['Metrics', 'YOLO Layer 52', 'YOLO Layer 26', 'YOLO Layer 13']]
        formats = {m: '%.6f' for m in metric_keys}
        for metric in metric_keys:
            row_metrics = [formats[metric] % ym.get(metric, 0) for ym in yolo_metrics]
            metric_table_data += [[metric, *row_metrics]]
        metric_table_data += [['total loss', '{:.6f}'.format(loss.item()), '', '']]
        # beautify log message
        metric_table = AsciiTable(
            metric_table_data,
            title='[Epoch {:d}/{:d}, Batch {:d}/{:d}]'.format(epoch, opt.num_epochs, i, len(dataloader)))
        metric_table.inner_footing_row_border = True
        logger.print_and_write('{}\n\n\n'.format(metric_table.table))
        
        if visualizer is not None:
            metrics_to_vis = []
            for j, ym in enumerate(yolo_metrics):
                for key, metric in ym.items():
                    if ym != 'grid_size':
                        metrics_to_vis += [('{}_yolo_layer_{}'.format(key, j), metric)]
                metrics_to_vis += [('total_loss', loss.item())]
            visualizer.plot_current_visuals(images_cpu, yolo_outputs)

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
            
