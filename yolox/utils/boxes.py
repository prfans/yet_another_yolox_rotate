#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
import numpy as np
import cv2

import torch
import torchvision

from shapely.geometry import Polygon
import shapely

__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
    "x1y1x2y2x3y3x4y4_to_cxcywha",
    "order_points",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]


def poly_ious(box1, box2):
    nBox = box2.shape[0]
    iou = torch.zeros(nBox)
    polygon1 = Polygon(box1.reshape(4,2)).convex_hull

    for i in range(0, nBox):
        polygon2 = Polygon(box2[i,:].reshape(4,2)).convex_hull
        if polygon1.intersects(polygon2):
            try:
                inter_area = polygon1.intersection(polygon2).area
                union_area = polygon1.union(polygon2).area
                iou[i] =  inter_area / union_area
            except shapely.geos.TopologicalError:
                print('shapely.geos.TopologicalError occured, iou set to 0')
                iou[i] = 0

    return iou


def _nms(detections, nms_thres=0.4):
    """ 非极大值抑制 """
    detections_t = detections.cpu()
    scores = detections_t[:, -2] * detections_t[:, -3]  # 计算目标的置信度

    # 目标排序
    _, conf_sort_index = torch.sort(scores, descending=True)
    detections_t = detections_t[conf_sort_index]

    # 提取目标的四顶点坐标
    all_pts = []
    for d in detections_t:
        x, y, w, h, angle = d[:5].numpy()
        if w < h:
            h, w = w, h
            angle += 90
        rotate_box = ((x, y), (w, h), angle)
        pts = cv2.boxPoints(rotate_box)
        pt4 = [pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1], pts[2, 0], pts[2, 1], pts[3, 0], pts[3, 1]]
        all_pts.append(pt4)
    all_pts = np.array(all_pts)

    # 非极大值抑制
    max_detections = []
    max_pts = []
    while detections_t.shape[0]:
        max_detections.append(detections_t[0])  # 最高置信度的bbox保存
        max_pts.append(all_pts[0])
        if len(detections_t) == 1:
            break
        ious = poly_ious(max_pts[-1], all_pts)  # 计算最高置信度bbox与其余bbox的iou
        detections_t = detections_t[ious < nms_thres]  # 去除最高置信度bbox周围的bbox
        all_pts = all_pts[ious < nms_thres]

    # 转成张量
    max_detections = [d.unsqueeze(0) for d in max_detections]
    max_detections = torch.cat(max_detections, dim=0)

    return max_detections


def non_max_suppression(detections, nms_thres=0.4, class_agnostic=False):
    """ non_max_suppression """
    # Detections ordered as (cx, cy, w, h, obj_conf, class_conf, class_pred, angle)
    detections = detections.cpu()
    if not class_agnostic:
        unique_labels = detections[:, -1].cpu().unique()
        dets = []
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            det_c = _nms(detections_class, nms_thres)
            dets.append(det_c)
        dets = torch.cat(dets, 0)
        return dets
    else:
        return _nms(detections, nms_thres)


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # [x y w h obj_conf cls... angle...]
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 6: 6 + num_classes], 1, keepdim=True)

        # 计算angle及置信度
        # angle_conf, angle_pred = torch.max(image_pred[:, 5 + num_classes:], 1, keepdim=True)
        angle_pred = (image_pred[:, -1]/np.pi*180).unsqueeze(1)
        # print('angle_pred: ', angle_pred.shape)
        # print('class_pred: ', class_pred.shape)image_pred
        image_pred[:, 4] = image_pred[:,4]/np.pi*180

        conf_mask = (image_pred[:, 5] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (cx, cy, w, h, obj_conf, class_conf, class_pred, angle)
        detections = torch.cat((image_pred[:, :6], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        detections = non_max_suppression(detections, nms_thre, class_agnostic)
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

        #exit(0)

    return output


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes


def order_points(pts):
    ''' sort rectangle points by clockwise '''
    def __sort_nx2(_pts):
        sort_x = _pts[np.argsort(_pts[:, 0]), :]
        Left = sort_x[:2, :]
        Right = sort_x[2:, :]
        # Left sort
        Left = Left[np.argsort(Left[:, 1])[::-1], :]
        # Right sort
        Right = Right[np.argsort(Right[:, 1]), :]

        return np.concatenate((Left, Right), axis=0)

    if pts.ndim == 2 and pts.shape[1] == 2: # (N, 2)
        return __sort_nx2(pts)
    elif pts.ndim == 2 and pts.shape[1] >= 8: # (N, 8)
        for n in range(pts.shape[0]):
            p = __sort_nx2(np.reshape(pts[n,:8], (4,2)))
            pts[n,:8] = np.reshape(p, pts[n,:8].shape)
        return pts
    elif pts.ndim == 1 and pts.shape[0] >= 8: # (8, )
        p = __sort_nx2(np.reshape(pts[:8], (4, 2)))
        pts[:8] = np.reshape(p, pts[:8].shape)
        return pts


def x1y1x2y2x3y3x4y4_to_cxcywha(bboxes):
    bboxes_t = []
    for bbox in bboxes:
        pts = np.reshape(np.float32(bbox), (len(bbox)//2,2))
        rect = cv2.minAreaRect(pts)

        # 通过实验发现，rotateRect规则：x轴按照顺时针旋转，
        # 与某条边平行时所转角度为旋转矩形角度，此边为旋转矩形的高
        # 那么另外一个边就是宽度，所以宽度和高度大小随机

        # 如果我们把长边固定座位宽，短边固定座位高
        # 如果需要交换宽高，则需要把角度+90度
        x = rect[0][0]
        y = rect[0][1]
        w = rect[1][0]
        h = rect[1][1]
        angle = math.ceil(rect[2])

        if w < h:
            h, w = w, h
            angle += 90

        angle = min(179, max(0, angle))
        angle = (angle / 180.0) * np.pi
        b = [x, y, w, h, angle]
        # print(b)
        bboxes_t.append(b)
    return np.array(bboxes_t, dtype=np.float32)


def x1y1x2y2x3y3x4y4_to_cxcywha_bak(bboxes):
    bboxes_t = []
    for bbox in bboxes:
        pts = np.reshape(np.float32(bbox), (len(bbox)//2,2))
        rect = cv2.minAreaRect(pts)
        x = rect[0][0]
        y = rect[0][1]
        w = rect[1][0]
        h = rect[1][1]
        angle = math.ceil(rect[2])
        angle = min(89, max(0, angle))
        angle = (angle / 180.0) * np.pi
        b = [x, y, w, h, angle]
        bboxes_t.append(b)
    return np.array(bboxes_t, dtype=np.float32)
