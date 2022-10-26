#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from yolox.utils import bboxes_iou, meshgrid

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

from .obb_iou_loss import cal_iou, cal_diou, cal_giou
from .poly_losses import PolyIoULoss, PolyGIOULoss, poly_iou_overlaps


class ComputeLoss(nn.Module):
    def __init__(
            self,
            num_classes,
            strides=[8, 16, 32],
            assignment="simOTA",
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        # self.iou_loss = IOUloss(reduction="none")
        self.iou_loss = PolyIoULoss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(strides)

        self.assignment_type = assignment

        self.assignments_method = None
        if self.assignment_type == "simOTA":
            self.assignments_method = self.get_assignments_simOTA
        elif self.assignment_type == "ATSS":
            self.assignments_method = self.get_assignments_ATSS
        elif self.assignment_type == "TOPK":
            self.assignments_method = self.get_assignments_TOPK
        elif self.assignment_type == "1V1":
            self.assignments_method = self.get_assignments_1V1
        else:
            raise RuntimeError("assignment type: {} is not supported. ".format(self.assignment_type))

    def forward(self, head_outputs, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (stride_this_level, x) in enumerate(
            zip(self.strides, head_outputs)
        ):
            reg_output = x[:, 0:5, :, :]
            obj_output = x[:, 5, :, :].unsqueeze(1)
            cls_output = x[:, 6: :, :]
    
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, head_outputs[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(head_outputs[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 5, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 5
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=outputs[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=outputs[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 6 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs_xy, outputs_wh, outputs_angles, outputs_obj, outputs_cls = torch.split(outputs,
                                                                                       [2, 2, 1, 1, self.num_classes],
                                                                                       dim=-1)
        outputs_xy = (outputs_xy + grids) * strides
        outputs_wh = torch.exp(outputs_wh) * strides
        outputs = torch.cat([outputs_xy, outputs_wh, outputs_angles, outputs_obj, outputs_cls], dim=-1)

        # outputs[..., :2] = (outputs[..., :2] + grids) * strides
        # outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        bbox_preds = outputs[:, :, :5]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 5].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 6:6 + self.num_classes]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        angle_targets = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 5))
                l1_target = outputs.new_zeros((0, 5))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                # 输入的label格式为 [class, x, y, w, h, angles...]
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:6]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.assignments_method(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    # if "CUDA out of memory. " not in str(e):
                    #    raise  # RuntimeError might not caused by CUDA OOM

                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.assignments_method(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                #print('fg_mask: ', fg_mask.shape, torch.sum(fg_mask), fg_mask.dtype)
                #print('gt_matched_classes: ', gt_matched_classes, gt_matched_classes.shape)
                #print('pred_ious_this_matching: ', pred_ious_this_matching, pred_ious_this_matching.shape)
                #print('matched_gt_inds: ', matched_gt_inds, matched_gt_inds.shape)
                #print(num_fg_img, end=' ')
                # logger.info(num_fg_img.item())


                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                # print(fg_mask.shape, fg_mask.dtype, fg_mask, torch.sum(fg_mask))
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 5)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 5)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 5)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)

        return l1_target

    def obb_overlaps(self, pred_, target_):

        """
        pred = torch.repeat_interleave(pred_, target_.shape[0], dim=0)
        target = torch.cat([target_ for i in range(pred_.shape[0])], dim=0)
        if pred.size(0) == 0 or target.size(0) == 0:
            return pred.sum() * 0.

        ious = cal_iou(pred.unsqueeze(0), target.unsqueeze(0))[0]

        return ious.view(size=(pred_.shape[0], target_.shape[0]))
        """
        return poly_iou_overlaps(pred_, target_)

    @torch.no_grad()
    def get_assignments_simOTA(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
            center_radius=2.5
        )

        if torch.sum(fg_mask).item() <= 10:
            print('Warning: fg_mask len is 0.')
            del fg_mask, is_in_boxes_and_center
            fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
                gt_bboxes_per_image,
                expanded_strides,
                x_shifts,
                y_shifts,
                total_num_anchors,
                num_gt,
                center_radius=5.0
            )
        assert fg_mask.any(), "Failed, len(fg_mask) == 0"

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)
        pair_wise_ious = self.obb_overlaps(gt_bboxes_per_image, bboxes_preds_per_image)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)

        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
            center_radius=2.5,
    ):
        expanded_strides_per_image = expanded_strides[0]  # # shape(1, n_anchors_all, 1)
        x_centers_per_image = (
            ((x_shifts[0] + 0.5) * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            ((y_shifts[0] + 0.5) * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_xy_per_image = gt_bboxes_per_image[:, None, 0:2]  # shape(num_gt, 1, 2)
        gt_wh_per_image = gt_bboxes_per_image[:, None, 2:4]  # shape(num_gt, 1, 2)
        gt_angles_per_image = gt_bboxes_per_image[:, 4, None]  # shape(num_gt, 1)

        grid_xy_per_image = torch.cat([x_centers_per_image.unsqueeze(2),
                                       y_centers_per_image.unsqueeze(2)], dim=-1)  # shape(1, n_anchor, 2)

        # in box
        Cos, Sin = torch.cos(gt_angles_per_image), torch.sin(gt_angles_per_image)  # shape(num_gt, 1)
        Matric = torch.stack([Cos, -Sin, Sin, Cos], dim=-1).repeat(1, total_num_anchors, 1, 1).view(num_gt,
                                                                                                    total_num_anchors,
                                                                                                    2, 2)
        offset = (grid_xy_per_image - gt_xy_per_image)[..., None]  # shape(num_gt, n_anchor, 2, 1)
        offset = torch.matmul(Matric, offset).squeeze_(-1)  # shape(n_gt, n_anchors, 2)

        b_lt = gt_wh_per_image / 2 + offset
        b_rb = gt_wh_per_image / 2 - offset
        bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)  # shape(n_gt, n_anchors, 4)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # shape(n_gt, n_anchors)
        is_in_boxes_all = is_in_boxes.sum(0) > 0  # shape(n_anchors)

        expanded_strides_per_image.unsqueeze_(0).unsqueeze_(2)

        # in center
        c_dist = center_radius * expanded_strides_per_image  # shape(1, n_anchors_all, 1)
        c_lt = grid_xy_per_image - (gt_xy_per_image - c_dist)
        c_rb = (gt_xy_per_image + c_dist) - grid_xy_per_image

        center_deltas = torch.cat([c_lt, c_rb], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # shape(num_gts, n_anchors_all)
        is_in_centers_all = is_in_centers.sum(dim=0) > 0  # shape(n_anchors_all)

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        # dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1, max=n_candidate_k)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    @torch.no_grad()
    def get_assignments_ATSS(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        device = bboxes_preds_per_image.device

        # 获取anchors
        expanded_strides_per_image = expanded_strides[0]  # # shape(1, n_anchors_all, 1)
        anchor_x = ((x_shifts[0] + 0.5) * expanded_strides_per_image)
        anchor_y = ((y_shifts[0] + 0.5) * expanded_strides_per_image)
        anchor_angle = torch.full_like(anchor_x, 0.0)
        anchors_per_image = torch.stack((anchor_x, anchor_y, expanded_strides_per_image, expanded_strides_per_image, anchor_angle), dim=1)
        anchor_points = torch.stack((anchor_x, anchor_y), dim=1)

        # anchor与gt的iou矩阵
        ious = self.obb_overlaps(anchors_per_image, gt_bboxes_per_image)

        # 获取strides
        unique_strides = torch.unique(expanded_strides)

        # 计算gt与bbox的L2距离
        gt_cx = gt_bboxes_per_image[:, 0]
        gt_cy = gt_bboxes_per_image[:, 1]
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        # 计算距离
        distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        
        # 以1-dis作为iou替代
        # ious = 1.0 - distances/distances.max()

        # 获取每个尺度的bbox数量
        num_anchors_per_level = []
        for stride in unique_strides:
            stride_mask = (expanded_strides == stride).squeeze(0)
            num_anchors_per_level.append(torch.sum(stride_mask))

        # 计算每个尺度上与gt的topk最小距离的bbox候选
        candidate_idxs = []
        star_idx = 0
        for level, stride in enumerate(unique_strides):
            end_idx = star_idx + num_anchors_per_level[level]
            distances_per_level = distances[star_idx:end_idx, :]
            topk = min(9, num_anchors_per_level[level])
            _, topk_idxs_per_level = distances_per_level.topk(topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + star_idx)
            star_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # 选择正样本，阈值使用iou的mean+std
        candidate_ious = ious[candidate_idxs, torch.arange(num_gt)]
        iou_mean_per_gt = candidate_ious.mean(0)
        iou_std_per_gt = candidate_ious.std(0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt
        is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

        # 最终的正样本：除了满足iou的阈值条件还需要预测的(cx cy)处于gt的框内
        _, is_in_boxes = self.get_anchor_in_bbox(gt_bboxes_per_image,
                                             expanded_strides,
                                             x_shifts,
                                             y_shifts,
                                             total_num_anchors,
                                             num_gt, )
        is_in_boxes = is_in_boxes.t()
        is_in_gts = is_in_boxes[candidate_idxs, torch.arange(num_gt)]
        is_pos = is_pos & is_in_gts  # 最终正样本

        # 如有重复，则每个bbox选择对应iou最高的gt
        INF = float('inf')
        ious_inf = torch.full_like(ious, -INF, device=device)
        ious_inf[candidate_idxs[is_pos], torch.arange(num_gt).repeat(candidate_idxs.shape[0], 1)[is_pos]] \
            = candidate_ious[is_pos]
        anchors_to_gt_values, anchors_to_gt_indexs = torch.max(ious_inf, dim=1)

        fg_mask = anchors_to_gt_values > -INF
        pred_ious_this_matching = anchors_to_gt_values[fg_mask]
        pred_ious_this_matching[:] = 1.0
        matched_gt_inds = anchors_to_gt_indexs[fg_mask]
        gt_matched_classes = gt_classes[anchors_to_gt_indexs][fg_mask]
        num_fg = torch.sum(fg_mask).item()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_distances(self, gt_xy_per_image, pred_xy_per_image):
        gt_xy = torch.repeat_interleave(gt_xy_per_image, pred_xy_per_image.shape[0], dim=0)
        pred_xy = pred_xy_per_image.repeat([gt_xy_per_image.shape[0], 1])
        distances = (gt_xy - pred_xy).pow(2).sum(-1).sqrt()
        distances = distances.reshape((gt_xy_per_image.shape[0], pred_xy_per_image.shape[0]))
        del gt_xy, pred_xy

        return distances

    def get_bbox_in_gt(
            self,
            gt_bboxes_per_image,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        pred_x = bboxes_preds_per_image[..., 0].repeat(num_gt, 1)
        pred_y = bboxes_preds_per_image[..., 1].repeat(num_gt, 1)
        pred_xy_per_image = torch.cat([pred_x.unsqueeze(2),
                                       pred_y.unsqueeze(2)], dim=-1)  # shape(1, n_anchor, 2)

        gt_xy_per_image = gt_bboxes_per_image[:, None, 0:2]  # shape(num_gt, 1, 2)
        gt_wh_per_image = gt_bboxes_per_image[:, None, 2:4]  # shape(num_gt, 1, 2)
        gt_angles_per_image = gt_bboxes_per_image[:, 4, None]  # shape(num_gt, 1)

        # in box
        Cos, Sin = torch.cos(gt_angles_per_image), torch.sin(gt_angles_per_image)  # shape(num_gt, 1)
        Matric = torch.stack([Cos, -Sin, Sin, Cos], dim=-1).repeat(1, total_num_anchors, 1, 1).view(num_gt,
                                                                                                    total_num_anchors,
                                                                                                    2, 2)
        offset = (pred_xy_per_image - gt_xy_per_image)[..., None]  # shape(num_gt, n_anchor, 2, 1)
        offset = torch.matmul(Matric, offset).squeeze_(-1)  # shape(n_gt, n_anchors, 2)

        b_lt = gt_wh_per_image / 2 + offset
        b_rb = gt_wh_per_image / 2 - offset
        bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)  # shape(n_gt, n_anchors, 4)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.01  # shape(n_gt, n_anchors)
        is_in_boxes_all = is_in_boxes.sum(0) > 0.01  # shape(n_anchors)

        return is_in_boxes_all, is_in_boxes

    def get_anchor_in_bbox(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
            center_radius=2.5,
    ):
        expanded_strides_per_image = expanded_strides[0]  # # shape(1, n_anchors_all, 1)
        x_centers_per_image = (
            ((x_shifts[0] + 0.5) * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            ((y_shifts[0] + 0.5) * expanded_strides_per_image)
                .unsqueeze(0)
                .repeat(num_gt, 1)
        )

        gt_xy_per_image = gt_bboxes_per_image[:, None, 0:2]  # shape(num_gt, 1, 2)
        gt_wh_per_image = gt_bboxes_per_image[:, None, 2:4]  # shape(num_gt, 1, 2)
        gt_angles_per_image = gt_bboxes_per_image[:, 4, None]  # shape(num_gt, 1)

        grid_xy_per_image = torch.cat([x_centers_per_image.unsqueeze(2),
                                       y_centers_per_image.unsqueeze(2)], dim=-1)  # shape(1, n_anchor, 2)

        # in box
        Cos, Sin = torch.cos(gt_angles_per_image), torch.sin(gt_angles_per_image)  # shape(num_gt, 1)
        Matric = torch.stack([Cos, -Sin, Sin, Cos], dim=-1).repeat(1, total_num_anchors, 1, 1).view(num_gt,
                                                                                                    total_num_anchors,
                                                                                                    2, 2)
        offset = (grid_xy_per_image - gt_xy_per_image)[..., None]  # shape(num_gt, n_anchor, 2, 1)
        offset = torch.matmul(Matric, offset).squeeze_(-1)  # shape(n_gt, n_anchors, 2)

        b_lt = gt_wh_per_image / 2 + offset
        b_rb = gt_wh_per_image / 2 - offset
        bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)  # shape(n_gt, n_anchors, 4)
        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # shape(n_gt, n_anchors)
        # is_in_boxes_all = is_in_boxes.sum(0) > 0  # shape(n_anchors)

        return bbox_deltas, is_in_boxes

    @torch.no_grad()
    def get_assignments_TOPK(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):
        # 获取anchors
        expanded_strides_per_image = expanded_strides[0]  # # shape(1, n_anchors_all, 1)
        anchor_x = ((x_shifts[0] + 0.5) * expanded_strides_per_image)
        anchor_y = ((y_shifts[0] + 0.5) * expanded_strides_per_image)
        anchor_angle = torch.full_like(anchor_x, 0.0)
        anchors_per_image = torch.stack((anchor_x, anchor_y, expanded_strides_per_image, expanded_strides_per_image, anchor_angle), dim=1)
        anchor_points = torch.stack((anchor_x, anchor_y), dim=1)

        # 获取strides
        unique_strides = torch.unique(expanded_strides)

        # 获取每个尺度的bbox数量
        num_anchors_per_level = []
        for stride in unique_strides:
            stride_mask = (expanded_strides == stride).squeeze(0)
            num_anchors_per_level.append(torch.sum(stride_mask))

        # gt中心
        gt_cx = gt_bboxes_per_image[:, 0]
        gt_cy = gt_bboxes_per_image[:, 1]
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        # bbox与gt的中心距离
        distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        distances = distances / distances.max() / 1000

        # bbox与gt的iou矩阵
        ious = self.obb_overlaps(anchors_per_image, gt_bboxes_per_image)

        # 预测点在gt内
        _, is_in_gt = self.get_anchor_in_bbox(gt_bboxes_per_image,
                                             expanded_strides,
                                             x_shifts,
                                             y_shifts,
                                             total_num_anchors,
                                             num_gt, )
        is_in_gt = is_in_gt.t()

        # topk
        is_pos = ious * False
        TOPK = 9
        for ng in range(num_gt):
            _, topk_idxs = (ious[:, ng] - distances[:, ng]).topk(TOPK, dim=0)
            is_in_gt_ng = is_in_gt[topk_idxs, ng]
            is_pos[topk_idxs[is_in_gt_ng == 1], ng] = True

        INF = float('inf')
        ious[is_pos == 0] = -INF
        anchors_to_gt_values, anchors_to_gt_indexs = ious.max(dim=1)

        # 正样本 负样本
        fg_mask = anchors_to_gt_values > -INF
        pred_ious_this_matching = anchors_to_gt_values[fg_mask]
        matched_gt_inds = anchors_to_gt_indexs[fg_mask]
        gt_matched_classes = gt_classes[anchors_to_gt_indexs][fg_mask]
        num_fg = torch.sum(fg_mask).item()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    @torch.no_grad()
    def get_assignments_1V1(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):
        # 获取strides
        unique_strides = torch.unique(expanded_strides)

        # 获取每个尺度的bbox数量
        num_anchors_per_level = []
        for stride in unique_strides:
            stride_mask = (expanded_strides == stride).squeeze(0)
            num_anchors_per_level.append(torch.sum(stride_mask))

        # gt的中心
        gt_cx = gt_bboxes_per_image[:, 0]
        gt_cy = gt_bboxes_per_image[:, 1]
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        # grid的中心
        expanded_strides_per_image = expanded_strides[0]  # # shape(1, n_anchors_all, 1)
        grid_x = ((x_shifts[0] + 0.5) * expanded_strides_per_image)
        grid_y = ((y_shifts[0] + 0.5) * expanded_strides_per_image)
        grid_xy_per_image = torch.stack((grid_x, grid_y), dim=1)

        # grid和gt的距离
        distances = (grid_xy_per_image[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        distances = distances / distances.max()

        # 以1-dis作为iou替代
        ious = 1.0 - distances

        #网格点在gt内
        _, is_in_gt = self.get_anchor_in_bbox(gt_bboxes_per_image,
                                             expanded_strides,
                                             x_shifts,
                                             y_shifts,
                                             total_num_anchors,
                                             num_gt, )
        is_in_gt = is_in_gt.t()

        # topk
        is_pos = ious * False
        TOPK = 9
        for ng in range(num_gt):
            _, topk_idxs = (-distances[:, ng]).topk(TOPK, dim=0)
            is_in_gt_ng = is_in_gt[topk_idxs, ng]
            is_pos[topk_idxs[is_in_gt_ng == 1], ng] = True

        INF = float('inf')
        ious[is_pos == 0] = -INF
        anchors_to_gt_values, anchors_to_gt_indexs = ious.max(dim=1)

        # 获取正样本 负样本
        fg_mask = anchors_to_gt_values > -INF
        pred_ious_this_matching = anchors_to_gt_values[fg_mask]
        matched_gt_inds = anchors_to_gt_indexs[fg_mask]
        gt_matched_classes = gt_classes[anchors_to_gt_indexs][fg_mask]
        num_fg = torch.sum(fg_mask).item()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )
