#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


class LOSSEvaluator:
    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
    ):
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
    ):
        # TODO half to amp_test
        tensor_type = torch.cuda.FloatTensor
        model = model.train()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        loss = 0.0
        for cur_iter, (imgs, targets, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                targets = targets.type(tensor_type)
                targets.requires_grad = False
                
                outputs = model(imgs, targets)
                
                loss += outputs["total_loss"]

        synchronize()
        return loss

