import  os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from terminaltables import AsciiTable

from utils.stats import (
    non_max_suppression, xywh2xyxy, 
    get_batch_statistics, ap_per_class, load_classe_names)


@torch.no_grad()
def test(model, dataloader, epoch, opt, test_logger, visualizer=None):
    labels = []
    sample_matrics = []
    for i, (images, targets) in enumerate(dataloader):
        if targets.size(0) == 0:
            continue
        
        batches_done = len(dataloader) * epoch + i
        if not opt.no_cuda:
            images_cpu = images.clone()
            model = model.to(opt.device)
            images = Variable(images.to(opt.device))
            if targets is not None:
                targets = Variable(targets.to(opt.device), requires_grad=False)

        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= opt.image_size

        detections = model.forward(images)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        sample_matrics += get_batch_statistics(detections, targets, iou_threshold=0.5)

        if visualizer is not None and not opt.no_vis_preds:
            visualizer.plot_predictions(images.cpu(), detections.cpu(), env='main') # plot prediction
        if visualizer is not None and not opt.no_vis_gt:
            visualizer.plot_ground_truth(images.cpu(), targets.cpu(), env='main') # plot ground truth
    
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_matrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    
    # logging
    metric_table_data = [
        ['Metrics', 'Value'], ['precision', precision.mean()], ['recall', recall.mean()], 
        ['f1', f1.mean()], ['mAP', AP.mean()]]
     
    metric_table = AsciiTable(
            metric_table_data,
            title='[Epoch {:d}/{:d}'.format(epoch, opt.num_epochs))
    print('{}\n\n\n'.format(metric_table.table))
    
    class_names = load_classe_names(opt.classname_path)
    for i, c in enumerate(ap_class):
        metric_table_data += [['AP-{}'.format(class_names[c]), AP[i]]]
    metric_table.table_data = metric_table_data
    test_logger.write('{}\n\n\n'.format(metric_table.table))

    vis.plot_metrics(images, detections)