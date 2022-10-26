#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .compute_loss import ComputeLoss


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None, compute_loss=None):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.compute_loss = compute_loss

    def forward(self, x, targets=None):
        fpn_outs = self.backbone(x)
        if self.head is not None:
            head_outs = self.head(fpn_outs)
        else:
            head_outs = fpn_outs

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.compute_loss(
                head_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.compute_loss(
                head_outs
            )

        return outputs
