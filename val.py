import  os
import torch
import torch.nn as nn 
from torch.autograd import Variable

from utils.stats import non_max_suppression, to_cpu, rescale_boxes, xywh2xyxy, get_batch_statistics, ap_per_class


@torch.no_grad()
def val(model, optimizer, dataloader, epoch, vis, opt):
    for i, (images, targets) in enumerate(dataloader):
        labels += targets[:, 1].tolist()
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= opt.img_size

        loss, yolo_outputs = model.forward(images, targets)
        outputs = non_max_suppression(outputs, opt.conf_thres, opt.nms_thres)

        error52 = model.yolo_layer52.metrics
        error26 = model.yolo_layer26.metrics
        error13 = model.yolo_layer13.metrics
        vis.print_current_losses([error52, error26, error13], epoch, i, len(dataloader))
        vis.plot_current_visuals(images_cpu, yolo_outputs)
        
        sample_matrics += get_batch_statistics(outputs, targets, iou_threshold=0.5)
    
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_matrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    print("Average Precision for Val:")
    for i, c in enumerate(ap_class):
        print('Class %s - AP: %.4f' % (class_names[c], AP[i]))

    print('mAP: %.4f' % (AP.mean()))