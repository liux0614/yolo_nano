import numpy as np 
import os
import visdom

import torch

from .stats import non_max_suppression, to_cpu, rescale_boxes
import random

from PIL import Image, ImageFont
from PIL.ImageDraw import Draw

import torchvision.transforms as transforms

def load_classe_names(classname_path):
    with open(classname_path, "r") as fp:
        class_names = fp.read().split("\n")
    return class_names

def tensor2im(tensor):
    img_np = tensor.cpu().float().numpy()
    img = np.clip((np.transpose(img_np, (1,2,0)) ), 0,1)
    return Image.fromarray(img, 'RGB')

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.ncols = opt.ncols
        self.classname_path = opt.classname_path
        self.class_names = load_classe_names(self.classname_path)
        assert len(self.class_names) == opt.num_classes
        
        self.log_path = os.path.join(self.opt.checkpoint_path, 'log.txt')

        self.viz = visdom.Visdom()


    def plot_target(self, imgs, targets):
        toImg = transforms.ToPILImage()
        for i in range(imgs.size(0)):
            img = toImg(imgs[i, ...])
            img = img.resize((540, 540))
            img_w, img_h = img.size
            ori_w, ori_h = img.size
            target = []
            for t in targets:
                if t[0] == i:
                    target.append(t)

            draw = Draw(img)
            for idx, cls_, cx, cy, w, h in target:
                if (cls_ == 0 or cls_ == 2):
                    continue
                x1 = int(float(cx) * img_w - float(w) * img_w / 2)
                y1 = int(float(cy) * img_h - float(h) * img_h / 2)
                x2 = int(float(cx) * img_w + float(w) * img_w / 2)
                y2 = int(float(cy) * img_h + float(h) * img_h / 2)
                draw.rectangle([(x1,y1), (x2,y2)], outline=(255,255,0))
            self.viz.image(np.array(img).transpose((2,0,1)), win=i)

    def plot_image(self, img):
        img = img.resize((480, 270))
        self.viz.image(np.array(img).transpose((2,0,1)), win=1)

    def print_current_losses(self, error_ret, epoch, cur_iter, total_iter):
        metrics = ['loss', 'x', 'y', 'w', 'h', 'conf','cls', 'cls_acc', 'recall50', 'recall75', 'precision', 'conf_obj', 'conf_noobj', 'grid_size']
        message = '\n----------[Epoch %d/%d, Batch %d/%d] -----------------\n' % (epoch, self.opt.num_epochs, cur_iter, total_iter)
        
        for key in metrics:
            message += '{:>10}\t{:>10.4f}\t{:10.4f}\t{:10.4f}\n'.format(key, error_ret[0][key], error_ret[1][key], error_ret[2][key])
        message += '------------------------------------------------------\n'

        print(message)
        with open(self.log_path, 'a') as log_file:
            log_file.write('%s\n' % message)

    # still has some bugs here
    def plot_current_visuals(self, imgs, detections):
        detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)
        toImg = transforms.ToPILImage()
        # we only show the image_batch[0]

        idx = []
        i = 0
        for detection in detections:
            if detection is not None:
                idx.append(i)
                i += 1

        if len(idx) == 0:
            self.viz.text('no bbox found with the conf_thres in %.2f' % (self.opt.conf_thres), win=1)
            return        

        for i in idx:
            img = toImg(imgs[i, ...])
            ori_w, ori_h = img.size 
            img = img.resize((270,270))
            detection = detections[i]
            w,h = img.size 
            draw = Draw(img)

            if detection is not None:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    x1 = float(x1) / ori_w * w
                    y1 = float(y1) / ori_h * h
                    x2 = float(x2) / ori_w * w 
                    y2 = float(y2) / ori_h * h 
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(int(x2), w)
                    y2 = min(int(y2), h)
                    draw.rectangle([(x1,y1), (x2,y2)], outline=(255,0,0))
                    #print(cls_pred)
                    name = self.class_names[int(cls_pred)]
                    name += "=%.4f" % float(cls_conf)
                    f = ImageFont.truetype("fonts-japanese-gothic.ttf", 15)
                    draw.text((x1,y1), name, 'blue', font=f)
                self.viz.image(np.array(img).transpose((2,0,1)), win=i+2)

    def plot_save_current_visual(self, img, detections):
        # plot the detections
        toImg = transforms.ToPILImage()
        img = toImg(img[0,...])
        ori_w, ori_h = img.size
        img = img.resize((540, 540))
        draw = Draw(img)
        w, h = img.size 
        for detection in detections:
            if detection is None:
                print("no detection is detected")
                return
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                x1 = float(x1) / ori_w * w;  x1 = max(0, int(x1))
                y1 = float(y1) / ori_h * h;  y1 = max(0, int(y1))
                x2 = float(x2) / ori_w * w;  x2 = min(int(x2), w)
                y2 = float(y2) / ori_h * h;  y2 = min(int(y2), h) 
                draw.rectangle([(x1, y1), (x2,y2)], outline=(255,255,0))
            self.viz.image(np.array(img).transpose((2,0,1)), win=1)