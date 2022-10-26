#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import random

import torch
import torch.distributed as dist

from yolox.exp import Exp as MyExp
from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, ComputeLoss
from yolov5.yolo import Model


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()


        # devices
        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        # model config file
        self.cfg_file = './yolov5/yolov5_repvgg.yaml'

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 6
        # factor of model depth
        self.depth = 1.0
        # factor of model width
        self.width = 1.0
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"
        # pos and neg sample selection method: "simOTA" or "ATSS" or "TOPK" or "1V1"
        self.assignment = "ATSS"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 8
        self.input_size = (320, 320)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir_train = r'train_data/'
        self.data_dir_test = r'test_data/'
        self.names_file = r'labels.names'

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 300
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 20
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 5
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (320, 320)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65


        # 此处增加新增的模型
        if getattr(self, "model", None) is None:
            example = torch.rand(size=(1, 3, *self.test_size), dtype=torch.float32)

            backbone = Model(self.cfg_file)

            output_backbone = backbone(example)
            print('====> backbone output: ')
            for o in output_backbone:
                print('  ', o.shape)
                # in_channels.append(o.shape[1])

            in_channels = [64, 64, 64]
            print('====> in_channels: \n  ', in_channels)

            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            compute_loss = ComputeLoss(self.num_classes, assignment=self.assignment)
            self.model = YOLOX(backbone, head, compute_loss)

            # import numpy as np
            # import cv2
            # pts = np.array([[0,0], [100, 0], [100, 50], [0, 50]])
            # rect = cv2.minAreaRect(pts)
            # print(rect)
            # exit(0)

            if True:
                #print(targets.shape)
                # print(data.shape)
                self.model.eval()
                outputs = self.model(example)
                print('====> model output: ', )
                for o in outputs:
                    print('  ', o.shape)
                # exit(0)

            # in_channels = [256, 512, 1024]
            # backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            # head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            # self.model = YOLOX(backbone, head)

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import (
            DOTADataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import wait_for_the_master

        with wait_for_the_master():
            dataset = DOTADataset(
                data_dir=self.data_dir_train,
                img_size=self.input_size,
                class_names=self.names_file,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import DOTADataset, TrainTransform

        valdataset = DOTADataset(
            data_dir=self.data_dir_test,
            img_size=self.test_size,
            class_names=self.names_file,
            preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=-1.0,
                    hsv_prob=self.hsv_prob),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import LOSSEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = LOSSEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator
