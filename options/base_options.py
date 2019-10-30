import argparse
import os
import torch

class BaseOptions():
    # this class defines options used during both training and test time
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # define the common options that are used in both train and test

        # the following is the options from orginal pytorch-yolov3
        self.parser.add_argument("--start_epochs", type=int, default=0, help="start from which epochs")
        self.parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
        self.parser.add_argument("--batch_size", type=int, default=2, help="size of each image batch")
        self.parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
        self.parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
        self.parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
        self.parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
        self.parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
        self.parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
        self.parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

        # add some model options
        #self.parser.add_argument("--model", type=str, default='drone_fire', help='choose which model to use')
        self.parser.add_argument("--checkpoint_dir", type=str, default='checkpoints', help='checkpoint_dir')
        self.parser.add_argument("--suffix", default='', type=str)
        self.parser.add_argument("--phase", type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument("--name", type=str, default='large_bbox')
        self.parser.add_argument("--gpu_ids", type=str, default='0,1')
        self.parser.add_argument("--num_classes", type=int, default=3)
        self.parser.add_argument("--display_freq", type=int, default=10, help='display the results in every # iterations')
        self.parser.add_argument("--ncols", type=int, default=5)
        self.parser.add_argument('--conf_thres', type=float, default=.8)
        self.parser.add_argument('--nms_thres', type=float, default=.5)
        self.parser.add_argument('--model', type=str, default='yolo_nano', help='yolo_nano')
        self.parser.add_argument("--dataset_mode", default="list", help="specify the type of custom dataset to create (crop | list | test | test_crop)")
        #self.parser.add_argument("--numAnchors", type=int, default=9, help='the # of anchors in each perdiction')

        self.initialized = True
        return self.parser

    def print_options(self, opt):
        # print the current options in the networks
        message = ''
        message += '------------------------ OPTIONS -----------------------------\n'
        for k,v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}\n'.format(str(k), str(v), comment)
        message += '------------------------  END   ------------------------------\n'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoint_dir, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        # parse the options, create checkpoint, and set up device
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()

        if self.opt.suffix:
            suffix = ('_' + self.opt.suffix.format(**vars(opt))) if self.opt.suffix != '' else ''
            self.opt.name = self.opt.name + suffix

        self.print_options(self.opt)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
        # set gpu ids
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        if len(self.opt.gpu_ids) > 0:
            self.opt.device = torch.device('cuda')
        else:
            self.opt.device = torch.device('cpu')

        #self.opt.Anchors = [[10,13], [16,30], [33,23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        self.opt.class_names = ['car', 'person', 'fire']

        return self.opt