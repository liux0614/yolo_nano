import os
import random
import visdom
import numpy as np 

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import torch
import torchvision.transforms as transforms

from .stats import load_classe_names
from transforms.bounding_box import BBox


class Visualizer():
    def __init__(self, opt, color_map=None):
        self.opt = opt
        self.class_names = load_classe_names(opt.classname_path)
        assert len(self.class_names) == opt.num_classes

        self.viz = visdom.Visdom()
        self.plots = {}

        self.color_map = plt.get_cmap('tab20b') if color_map is None else color_map
        self.colors = [self.color_map(i) for i in np.linspace(0, 1, opt.num_classes)]

    def plot_metrics(self, metrics, x, env='metrics'):
        for (name, y) in metrics:
            if name not in self.plots:
                self.plots[name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=env, opts=dict(
                    title=name,
                    xlabel='Batches',
                    ylabel=name
            ))
            else:
                self.viz.line(X=np.array([x]), Y=np.array([y]), win=self.plots[name], env=env, update='append')

    def clean_matplot(self):
        fig, ax = plt.subplots(1)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.tight_layout(pad = 0)
        return fig, ax

    def plot_ground_truth(self, images, bboxes, env='main'):
        plt.figure()

        for i in range(images.size(0)):
            image_i = images[i, ...].transpose(0, 1).transpose(1, 2)
            image_size = image_i.size()[:2]
            bboxes_image_i = bboxes[ bboxes[:, 0] == i ]
            if bboxes_image_i.size(0) == 0:
                continue
            
            bboxes_image_i = BBox.from_yolo(bboxes_image_i[:, 1:], image_size)
            bboxes_image_i = bboxes_image_i.to_tensor(mode='coco')
            # print(f"plot: {bboxes_image_i}")

            unique_labels = bboxes_image_i[:, 0].unique()
            num_cls_preds = len(unique_labels)
            bbox_colors = random.sample(self.colors, num_cls_preds)
            
            fig, ax = self.clean_matplot()
            ax.imshow(image_i.numpy())
            for cls_pred, xmin, ymin, box_w, box_h in bboxes_image_i:
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                bbox = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                ax.add_patch(bbox)
                plt.text(
                    xmin,
                    ymin,
                    s = self.class_names[int(cls_pred)],
                    color = 'white',
                    verticalalignment = 'top',
                    bbox = {'color': color, 'pad': 0},
                )
            
            name = 'gt/image_{}'.format(i)
            if name not in self.plots:
                self.plots[name] = self.viz.matplot(plt, env=env, opts=dict(title=name))
            else:
                self.viz.matplot(plt, win=self.plots[name], env=env, opts=dict(title=name))

            plt.close()
            # By default, only the first image in a batch is visualized due to
            # performance issue caused by ``visdom.matplot``
            if not self.opt.vis_all_images:
                break


    def plot_predictions(self, images, detections, env='main'):
        plt.figure()

        idx = []
        i = 0
        for detection in detections:
            if detection is not None:
                idx.append(i)
                i += 1

        for i in idx:
            image_i = images[i, ...].transpose(0, 1).transpose(1, 2)
            detection = detections[i]
            
            fig, ax = self.clean_matplot()
            ax.imshow(image_i.numpy())

            name = 'preds/image_{}'.format(i)
            if detection is None:
                text = 'No bounding box found with confidence threshold - {} %.2f'.format(self.opt.conf_thres)
                if name not in self.plots:
                    self.plots[name] = self.viz.text(text, env=env, opts=dict(title=name))
                else:
                    self.viz.text(text, win=self.plots[name], env=env, opts=dict(title=name))
                continue

            unique_labels = detection[:, -1].unique()
            num_cls_preds = len(unique_labels)
            bbox_colors = random.sample(self.colors, num_cls_preds)

            for xmin, ymin, xmax, ymax, conf, cls_conf, cls_pred in detection:
                box_w = xmax - xmin
                box_h = ymax - ymin
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                bbox = patches.Rectangle((xmin, ymin), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                ax.add_patch(bbox)
                plt.text(
                    xmin,
                    ymin,
                    s = '{} {:.4f}'.format(self.class_names[int(cls_pred)], cls_conf),
                    color = 'white',
                    verticalalignment = 'top',
                    bbox = {'color': color, 'pad': 0},
                )
            
            if name not in self.plots:
                self.plots[name] = self.viz.matplot(plt, env=env, opts=dict(title=name))
            else:
                self.viz.matplot(plt, win=self.plots[name], env=env, opts=dict(title=name))

            plt.close()
            # By default, only the first image in a batch is visualized due to
            # performance issue caused by ``visdom.matplot``
            if not self.opt.vis_all_images:
                break