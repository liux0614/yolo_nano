import os
import json
import torch
import argparse

class Opt():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # project root, dataset, checkpoint resume and pretrained model path
        self.parser.add_argument("--project_root", type=str, default=".", help="root directory path of project")
        self.parser.add_argument("--dataset_path", type=str, default="datasets/coco/jpg", help="directory path of dataset")
        self.parser.add_argument("--annotation_path", type=str, default="datasets/coco/annotation", help="file path of annotations")
        self.parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="directory path of checkpoints")
        self.parser.add_argument("--resume_path", type=str, default="", help="save data (.pth) of previous training")
        # self.parser.add_argument("--pretrain_path", type=str, default="", help="path of pretrain model (.pth)")

        # common options that are used in both train and test
        self.parser.add_argument("--manual_seed", type=int, default=42, help="manual_seed of pytorch")
        self.parser.add_argument("--no_cuda", action="store_true", help="if true, cuda is not used")
        self.parser.set_defaults(no_cuda=False)

        self.parser.add_argument("--num_threads", type=int, default=8, help="# of cpu threads to use for batch generation")
        self.parser.add_argument("--dataset", default="coco", help="specify the type of custom dataset to create")
        self.parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
        self.parser.add_argument("--val_interval", type=int, default=5, help="evaluation every 5 epochs")

        self.parser.add_argument("--model", type=str, default="yolo_nano", help="choose which model to use")
        self.parser.add_argument("--image_size", type=int, default=416, help="size of image")
        self.parser.add_argument("--num_classes", type=int, default=80, help="# of classes of the dataset")
        self.parser.add_argument('--num_epochs', type=int, default=20, help='# of epochs')
        self.parser.add_argument('--begin_epoch', type=int, default=0, help='# of epochs')
        self.parser.add_argument("--batch_size", type=int, default=8, help="batch size")
        self.parser.add_argument('--gradient_accumulations', type=int, default=1, help="number of gradient accums before step")

        self.parser.add_argument("--optimizer", type=str, default="Adam", help="optimizer (Adam | SGD | AdaBound)")
        self.parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
        self.parser.add_argument('--momentum', type=float, default=0.9, help="momentum for optimizer")
        self.parser.add_argument('--weight_decay', type=float, default=1e-3, help="weight_decay for optimizer")
        self.parser.add_argument('--final_lr', type=float, default=0.1, help="final learning rate used by AdaBound optimizer")
        
        # object detection options
        self.parser.add_argument("--conf_thres", type=float, default=.5)
        self.parser.add_argument("--nms_thres", type=float, default=.5)
        
        self.parser.add_argument("--no_multi_scale", action="store_true", help="if true, no multi-scale augmentation")
        self.parser.set_defaults(no_multi_scale=False)
        self.parser.add_argument("--no_pad2square", action="store_true", help="if true, no pad to square augmentation")
        self.parser.set_defaults(no_pad2square=False)
        # self.parser.add_argument("--no_crop", action="store_true", help="if true, no random crop augmentation")
        # self.parser.set_defaults(no_crop=False)
        # self.parser.add_argument('--crop_size', type=int, default=540, help="crop the images to ``crop_size``")
        self.parser.add_argument("--no_hflip", action="store_true", help="if true, no random horizontal-flip augmentation")
        self.parser.set_defaults(no_hflip=False)
        self.parser.add_argument('--hflip_prob', type=float, default=.5, help="the probability of flipping the image and bboxes horozontally")

        self.parser.add_argument("--no_train", action="store_true", help="training")
        self.parser.set_defaults(no_train=False)
        self.parser.add_argument("--no_val", action="store_true", help="validation")
        self.parser.set_defaults(no_val=False)
        self.parser.add_argument("--test", default=False, help="test")
        
        # visualizer
        self.parser.add_argument("--no_vis", action="store_true", help="if true, no visualization")
        self.parser.set_defaults(no_vis=False)
        self.parser.add_argument("--no_vis_gt", action="store_true", help="if true, no visualization for ground truth")
        self.parser.set_defaults(no_vis_gt=False)
        self.parser.add_argument("--no_vis_preds", action="store_true", help="if true, no visualization for predictions")
        self.parser.set_defaults(no_vis_preds=False)
        self.parser.add_argument("--vis_all_images", action="store_true", help="if true, visualize all images in a batch")
        self.parser.set_defaults(vis_all_images=False)
        self.parser.add_argument("--classname_path", type=str, default="datasets/coco.names", help="file path of classnames for visualizer")
        self.parser.add_argument("--print_options", default=True, help="print options or not")

        self.initialized = True

    def print_options(self):
        message = ''
        message += '------------------------ OPTIONS -----------------------------\n'
        for k,v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}\n'.format(str(k), str(v), comment)
        message += '------------------------  END   ------------------------------\n'
        print(message)

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        if self.opt.project_root != '':
            self.opt.dataset_path = os.path.join(self.opt.project_root, self.opt.dataset_path)
            self.opt.checkpoint_path = os.path.join(self.opt.project_root, self.opt.checkpoint_path)
            if self.opt.resume_path:
                self.opt.resume_path = os.path.join(self.opt.project_root, self.opt.resume_path)
            # if self.opt.pretrain_path:
            #     self.opt.pretrain_path = os.path.join(self.opt.project_root, self.opt.pretrain_path)

        os.makedirs(self.opt.checkpoint_path, exist_ok=True)
        
        with open(os.path.join(self.opt.checkpoint_path, 'opts.json'), 'w') as opt_file:
            json.dump(vars(self.opt), opt_file)

        self.opt.device = torch.device('cpu') if self.opt.no_cuda else torch.device('cuda')

        if self.opt.print_options:
            self.print_options()

        return self.opt