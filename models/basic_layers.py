import torch
import torch.nn as nn
from utils.stats import build_targets, to_cpu, non_max_suppression


def conv1x1(input_channels, output_channels, stride=1, bn=True):
    # 1x1 convolution without padding
    if bn == True:
        return nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=1,
                stride=stride, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )
    else:
        return nn.Conv2d(
                input_channels, output_channels, kernel_size=1,
                stride=stride, bias=False)


def conv3x3(input_channels, output_channels, stride=1, bn=True):
    # 3x3 convolution with padding=1
    if bn == True:
        return nn.Sequential(
            nn.Conv2d(
                input_channels, output_channels, kernel_size=3,
                stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU6(inplace=True)
        )
    else:
        nn.Conv2d(
                input_channels, output_channels, kernel_size=3,
                stride=stride, padding=1, bias=False)

def sepconv3x3(input_channels, output_channels, stride=1, expand_ratio=1):
    return nn.Sequential(
        # pw
        nn.Conv2d(
            input_channels, input_channels * expand_ratio,
            kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(input_channels * expand_ratio),
        nn.ReLU6(inplace=True),
        # dw
        nn.Conv2d(
            input_channels * expand_ratio, input_channels * expand_ratio, kernel_size=3, 
            stride=stride, padding=1, groups=input_channels * expand_ratio, bias=False),
        nn.BatchNorm2d(input_channels * expand_ratio),
        nn.ReLU6(inplace=True),
        # pw-linear
        nn.Conv2d(
            input_channels * expand_ratio, output_channels,
            kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(output_channels)
    )

class EP(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(EP, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        self.sepconv = sepconv3x3(input_channels, output_channels, stride=stride)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.sepconv(x)
        
        return self.sepconv(x)

class PEP(nn.Module):
    def __init__(self, input_channels, output_channels, x, stride=1):
        super(PEP, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.use_res_connect = self.stride == 1 and input_channels == output_channels

        self.conv = conv1x1(input_channels, x)
        self.sepconv = sepconv3x3(x, output_channels, stride=stride)
        
    def forward(self, x):        
        out = self.conv(x)
        out = self.sepconv(out)
        if self.use_res_connect:
            return out + x

        return out


class FCA(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(FCA, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        hidden_channels = channels // reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_channels, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(hidden_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out

# From https://github.com/eriklindernoren/PyTorch-YOLOv3
class YOLOLayer(nn.Module):
    # detection layer
    def __init__(self, anchors, num_classes, img_dim=416):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = .5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        self.grid_size = 0

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor([(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        self.img_dim = img_dim

        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes+5, grid_size, grid_size)
            .permute(0,1,3,4,2)
            .contiguous()
        )

        x = torch.sigmoid(prediction[..., 0]) # center x
        y = torch.sigmoid(prediction[..., 1]) # center y
        w = prediction[..., 2] # width
        h = prediction[..., 3] # Height
        pred_conf = torch.sigmoid(prediction[..., 4]) # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:]) # Cls Pred

        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h
        #print(pred_boxes.size())
        #print(pred_boxes.view(num_samples, -1, 4).size())

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if targets is None:
            return output, 0
        else:
            iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                pred_boxes=pred_boxes,
                pred_cls=pred_cls,
                target=targets,
                anchors=self.scaled_anchors,
                ignore_thres=self.ignore_thres,
            )

            # Loss : Mask outputs to ignore non-existing objects (except with conf. loss)
            loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
            loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
            loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
            loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
            #print(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
            loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
            loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
            loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])
            total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            # Metrics
            cls_acc = 100 * class_mask[obj_mask].mean()
            conf_obj = pred_conf[obj_mask].mean()
            conf_noobj = pred_conf[noobj_mask].mean()
            conf50 = (pred_conf > 0.5).float()
            iou50 = (iou_scores > 0.5).float()
            iou75 = (iou_scores > 0.75).float()
            detected_mask = conf50 * class_mask * tconf
            precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
            recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
            recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

            self.metrics = {
                "loss": to_cpu(total_loss).item(),
                "x": to_cpu(loss_x).item(),
                "y": to_cpu(loss_y).item(),
                "w": to_cpu(loss_w).item(),
                "h": to_cpu(loss_h).item(),
                "conf": to_cpu(loss_conf).item(),
                "cls": to_cpu(loss_cls).item(),
                "cls_acc": to_cpu(cls_acc).item(),
                "recall50": to_cpu(recall50).item(),
                "recall75": to_cpu(recall75).item(),
                "precision": to_cpu(precision).item(),
                "conf_obj": to_cpu(conf_obj).item(),
                "conf_noobj": to_cpu(conf_noobj).item(),
                "grid_size": grid_size,
            }
            return output, total_loss