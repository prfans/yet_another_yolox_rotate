#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import cv2
import numpy as np

from .datasets_wrapper import Dataset
from .get_im_list import get_im_list
from ...utils import order_points


class DOTADataset(Dataset):
    """
    DOTA dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        img_size=(416, 416),
        class_names=None,
        preproc=None,
        cache=False,
    ):
        """
        DOTA dataset initialization.
        """
        super().__init__(img_size)
        if data_dir is None:
            assert 0, "Error: data_dir is None"
        self.data_dir = data_dir

        # 加载所有图像文件及对应dota标注txt文件
        img_file_all = get_im_list(self.data_dir, '.bmp')
        img_file_all += get_im_list(self.data_dir, '.png')
        img_file_all += get_im_list(self.data_dir, '.jpg')
        self.img_files = []
        self.label_files = []
        for im_file in img_file_all:
            label_file = im_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.bmp', '.txt')
            if os.path.exists(label_file) and os.path.exists(im_file):
                self.img_files.append(im_file)
                self.label_files.append(label_file)

        # 加载names文件
        with open(class_names, 'r') as f:
            self.class_names = f.readlines()
        self.class_names = [name.rstrip() for name in self.class_names]

        # 预先加载图像加速
        self.cache = cache
        self.imgs = None
        if self.cache:
            self.imgs = [None] * len(self.img_files)
            self.imgs_shape = [None] * len(self.img_files)
            for i in range(len(self.img_files)):
                self.imgs[i], self.imgs_shape[i] = self.load_resized_img(i)

        self.img_size = img_size
        self.preproc = preproc

    def __len__(self):
        return len(self.img_files)

    def __del__(self):
        del self.imgs

    def load_anno_from_ids(self, index, shape=None):
        label_path = self.label_files[index]

        if shape is None:
            shape = self.imgs_shape[index]
        height, width = shape[:2]

        # Load labels
        if os.path.isfile(label_path):
            with open(label_path, 'r') as fd:
                lines = fd.readlines()
                labels0 = np.array([x.split() for x in lines])
            annotations = np.ones((len(labels0), 9))
            annotations[:, 8] = np.array([self.class_names.index(name) for name in labels0[:, 0]])
            annotations[:, 0] = labels0[:, 1].astype('float32')
            annotations[:, 1] = labels0[:, 2].astype('float32')
            annotations[:, 2] = labels0[:, 3].astype('float32')
            annotations[:, 3] = labels0[:, 4].astype('float32')
            annotations[:, 4] = labels0[:, 5].astype('float32')
            annotations[:, 5] = labels0[:, 6].astype('float32')
            annotations[:, 6] = labels0[:, 7].astype('float32')
            annotations[:, 7] = labels0[:, 8].astype('float32')
        else:
            annotations = np.array([])

        objs = []
        for obj in annotations:
            obj_pts = np.array(obj[:8], dtype=np.float32)
            obj_pts = np.reshape(obj_pts, (4, 2))

            rect = cv2.minAreaRect(obj_pts)
            p4 = cv2.boxPoints(rect)
            p4 = order_points(p4)

            # x = rect[0][0]
            # y = rect[0][1]
            w = rect[1][1]
            h = rect[1][0]
            # angle = rect[2]
            if w > 0 and h > 0:
                pt_flat = [p4[0,0],p4[0,1],p4[1,0],p4[1,1],p4[2,0],p4[2,1],p4[3,0],p4[3,1]]
                objs.append(pt_flat)

        num_objs = len(objs)

        res = np.zeros((num_objs, 9))

        for ix, obj in enumerate(objs):
            cls = annotations[ix,8]
            res[ix, 0:8] = obj
            res[ix, 8] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :8] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = self.img_files[index]

        return (res, img_info, resized_info, file_name)

    # 判断图像及标注文件是否存在
    def has_anno(self, index):
        return (os.path.exists(self.label_files[index])
                and os.path.exists(self.img_files[index]))

    def load_resized_img(self, index):
        img = self.load_image(index)
        shape = img.shape
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img, shape

    def load_image(self, index):
        img_file = self.img_files[index]
        img = cv2.imread(img_file)
        assert img is not None, f"file named not found"

        return img

    def pull_item(self, index):
        if self.imgs is not None:
            res, img_info, resized_info, _ = self.load_anno_from_ids(index)
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img, shape= self.load_resized_img(index)
            res, img_info, resized_info, _ = self.load_anno_from_ids(index, shape=shape)

        return img, res.copy(), img_info, np.array([index])

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id = self.pull_item(index) # [x1 y1 x2 y2 x3 y3 x4 y4 c]

        if self.preproc is not None: # [cx cy w h c angle]
            img, target = self.preproc(img, target, self.input_dim)

        return img, target, img_info, img_id
