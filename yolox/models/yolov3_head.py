#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import numpy as np

from .network_blocks import BaseConv, DWConv


class YOLOV3Head(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.stems = nn.ModuleList()
        self.mid_convs = nn.ModuleList()
        self.head_preds = nn.ModuleList()

        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(in_channels[i] * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )

            self.mid_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(in_channels[i] * width),
                            out_channels=int(in_channels[i] * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(in_channels[i] * width),
                            out_channels=int(in_channels[i] * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )

            self.head_preds.append(
                nn.Conv2d(
                    in_channels=int(in_channels[i] * width),
                    out_channels=self.n_anchors * (5  + 1 + self.num_classes),
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

    def initialize_biases(self, prior_prob):

        for conv in self.head_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin):
        outputs = []

        for k, (stem, mid_conv, head_conv, x) in enumerate(
            zip(self.stems, self.mid_convs, self.head_preds, xin)
        ):
            outs = stem(x)
            outs = mid_conv(outs)
            outs = head_conv(outs)

            outputs.append(outs)

        return outputs
