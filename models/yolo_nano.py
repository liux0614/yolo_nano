import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_layers import conv1x1, conv3x3, EP, PEP, FCA, YOLOLayer
from utils.stats import build_targets, to_cpu, non_max_suppression


class YOLONano(nn.Module):
    def __init__(self, num_classes, image_size):
        super(YOLONano, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_anchors = 3
        self.yolo_channels = (self.num_classes + 5) * self.num_anchors
        
        anchors52 = [[10,13], [16,30], [33,23]] # 52x52
        anchors26 = [[30,61], [62,45], [59,119]] # 26x26
        anchors13 = [[116,90], [156,198], [373,326]] # 13x13
        
        # image:  416x416x3
        self.conv1 = conv3x3(3, 12, stride=1) # output: 416x416x12
        self.conv2 = conv3x3(12, 24, stride=2) # output: 208x208x24
        self.pep1 = PEP(24, 24, 7, stride=1) # output: 208x208x24
        self.ep1 = EP(24, 70, stride=2) # output: 104x104x70
        self.pep2 = PEP(70, 70, 25, stride=1) # output: 104x104x70
        self.pep3 = PEP(70, 70, 24, stride=1) # output: 104x104x70
        self.ep2 = EP(70, 150, stride=2) # output: 52x52x150
        self.pep4 = PEP(150, 150, 56, stride=1) # output: 52x52x150
        self.conv3 = conv1x1(150, 150, stride=1) # output: 52x52x150
        self.fca1 = FCA(150, 8) # output: 52x52x150
        self.pep5 = PEP(150, 150, 73, stride=1) # output: 52x52x150
        self.pep6 = PEP(150, 150, 71, stride=1) # output: 52x52x150
        
        self.pep7 = PEP(150, 150, 75, stride=1) # output: 52x52x150
        self.ep3 = EP(150, 325, stride=2) # output: 26x26x325
        self.pep8 = PEP(325, 325, 132, stride=1) # output: 26x26x325
        self.pep9 = PEP(325, 325, 124, stride=1) # output: 26x26x325
        self.pep10 = PEP(325, 325, 141, stride=1) # output: 26x26x325
        self.pep11 = PEP(325, 325, 140, stride=1) # output: 26x26x325
        self.pep12 = PEP(325, 325, 137, stride=1) # output: 26x26x325
        self.pep13 = PEP(325, 325, 135, stride=1) # output: 26x26x325
        self.pep14 = PEP(325, 325, 133, stride=1) # output: 26x26x325
        
        self.pep15 = PEP(325, 325, 140, stride=1) # output: 26x26x325
        self.ep4 = EP(325, 545, stride=2) # output: 13x13x545
        self.pep16 = PEP(545, 545, 276, stride=1) # output: 13x13x545
        self.conv4 = conv1x1(545, 230, stride=1) # output: 13x13x230
        self.ep5 = EP(230, 489, stride=1) # output: 13x13x489
        self.pep17 = PEP(489, 469, 213, stride=1) # output: 13x13x469
        
        self.conv5 = conv1x1(469, 189, stride=1) # output: 13x13x189
        self.conv6 = conv1x1(189, 105, stride=1) # output: 13x13x105
        # upsampling conv6 to 26x26x105
        # concatenating [conv6, pep15] -> pep18 (26x26x430)
        self.pep18 = PEP(430, 325, 113, stride=1) # output: 26x26x325
        self.pep19 = PEP(325, 207, 99, stride=1) # output: 26x26x325
        
        self.conv7 = conv1x1(207, 98, stride=1) # output: 26x26x98
        self.conv8 = conv1x1(98, 47, stride=1) # output: 26x26x47
        # upsampling conv8 to 52x52x47
        # concatenating [conv8, pep7] -> pep20 (52x52x197)
        self.pep20 = PEP(197, 122, 58, stride=1) # output: 52x52x122
        self.pep21 = PEP(122, 87, 52, stride=1) # output: 52x52x87
        self.pep22 = PEP(87, 93, 47, stride=1) # output: 52x52x93
        self.conv9 = conv1x1(93, self.yolo_channels, stride=1, bn=False) # output: 52x52x yolo_channels
        self.yolo_layer52 = YOLOLayer(anchors52, num_classes, img_dim=image_size)

        # conv7 -> ep6
        self.ep6 = EP(98, 183, stride=1) # output: 26x26x183
        self.conv10 = conv1x1(183, self.yolo_channels, stride=1, bn=False) # output: 26x26x yolo_channels
        self.yolo_layer26 = YOLOLayer(anchors26, num_classes, img_dim=image_size)

        # conv5 -> ep7
        self.ep7 = EP(189, 462, stride=1) # output: 13x13x462
        self.conv11 = conv1x1(462, self.yolo_channels, stride=1, bn=False) # output: 13x13x yolo_channels
        self.yolo_layer13 = YOLOLayer(anchors13, num_classes, img_dim=image_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.xavier_normal_(m.weight, gain=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, x, targets=None):
        loss = 0
        yolo_outputs = []
        image_size = x.size(2)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pep1(out)
        out = self.ep1(out)
        out = self.pep2(out)
        out = self.pep3(out)
        out = self.ep2(out)
        out = self.pep4(out)
        out = self.conv3(out)
        out = self.fca1(out)
        out = self.pep5(out)
        out = self.pep6(out)
        
        out_pep7 = self.pep7(out)
        out = self.ep3(out_pep7)
        out = self.pep8(out)
        out = self.pep9(out)
        out = self.pep10(out)
        out = self.pep11(out)
        out = self.pep12(out)
        out = self.pep13(out)
        out = self.pep14(out)

        out_pep15 = self.pep15(out)
        out = self.ep4(out_pep15)
        out = self.pep16(out)
        out = self.conv4(out)
        out = self.ep5(out)
        out = self.pep17(out)

        out_conv5 = self.conv5(out)
        out = F.interpolate(self.conv6(out_conv5), scale_factor=2)
        out = torch.cat([out, out_pep15], dim=1)
        out = self.pep18(out)
        out = self.pep19(out)
        
        out_conv7 = self.conv7(out)
        out = F.interpolate(self.conv8(out_conv7), scale_factor=2)
        out = torch.cat([out, out_pep7], dim=1)
        out = self.pep20(out)
        out = self.pep21(out)
        out = self.pep22(out)
        out_conv9 = self.conv9(out)
        temp, layer_loss = self.yolo_layer52(out_conv9, targets, image_size)
        loss += layer_loss
        yolo_outputs.append(temp)

        out = self.ep6(out_conv7)
        out_conv10 = self.conv10(out)
        temp, layer_loss = self.yolo_layer26(out_conv10, targets, image_size)
        loss += layer_loss
        yolo_outputs.append(temp)

        out = self.ep7(out_conv5)
        out_conv11 = self.conv11(out)
        temp, layer_loss = self.yolo_layer13(out_conv11, targets, image_size)
        loss += layer_loss
        yolo_outputs.append(temp)

        yolo_outputs = to_cpu(torch.cat(yolo_outputs, 1))

        return yolo_outputs if targets is None else (loss, yolo_outputs)

    def name(self):
        return "YoloNano"