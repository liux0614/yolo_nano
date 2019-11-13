import  os
import torch
import torch.nn as nn 
from torch.autograd import Variable

from terminaltables import AsciiTable


def train(model, optimizer, dataloader, epoch, opt, logger, visualizer=None):
    for i, (images, targets) in enumerate(dataloader):
        # targets: [idx, class_id, x, y, h, w] in yolo format
        # idx is used to associate the bounding boxes with its image
        # skip images without bounding boxes (mainly because coco has unlabelled images) 
        if targets.size(0) == 0:
            continue
        
        batches_done = len(dataloader) * epoch + i
        if not opt.no_cuda:
            images_cpu = images.clone()
            model = model.to(opt.device)
            images = Variable(images.to(opt.device))
            if targets is not None:
                targets = Variable(targets.to(opt.device), requires_grad=False)
        
        loss, detections = model.forward(images, targets)
        loss.backward()

        if batches_done % opt.gradient_accumulations == 0:
            optimizer.step()
            optimizer.zero_grad()

        # logging
        metric_keys = model.yolo_layer52.metrics.keys()
        yolo_metrics = [model.yolo_layer52.metrics, model.yolo_layer26.metrics, model.yolo_layer13.metrics]
        metric_table_data = [['Metrics', 'YOLO Layer 0', 'YOLO Layer 1', 'YOLO Layer 2']]
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
        
        if visualizer is not None and not opt.no_vis_preds:
            visualizer.plot_predictions(images.cpu(), detections.cpu(), env='train') # plot prediction
        if visualizer is not None and not opt.no_vis_gt:
            visualizer.plot_ground_truth(images.cpu(), targets.cpu(), env='train') # plot ground truth
        
        metrics_to_vis = []
        # uncomment code below to plot the metrics of each YOLO layer
        # for j, ym in enumerate(yolo_metrics):
        #     for key, metric in ym.items():
        #         if key != 'grid_size':
        #             metrics_to_vis += [('{}_yolo_layer_{}'.format(key, j), metric)]
        metrics_to_vis += [('total_loss', loss.item())]
        visualizer.plot_metrics(metrics_to_vis, batches_done, env='loss')

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
            
