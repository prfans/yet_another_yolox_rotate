import torch
import BboxToolkit as bt

import cv2
import numpy as np
import torch
from torch.autograd import Function
from torch.autograd import Variable
from torch.autograd import gradcheck

"""
class OBBOverlaps():

    def __call__(self, bboxes1, bboxes2, mode='iou'):
        bboxes1 = bboxes1.float()
        bboxes2 = bboxes2.float()
        assert bboxes2.shape[-1] in [0, 5, 6]
        assert bboxes1.shape[-1] in [0, 5, 6]
        
        if bboxes1.shape[-1] == 6:
            bboxes1 = bboxes1[..., :5]
        if bboxes2.shape[-1] == 6:
            bboxes2 = bboxes2[..., :5]
        return obb_overlaps(bboxes1, bboxes2, mode)

    def __repr__(self):
        repr_str = self.__class__.__name__ + '()'
        return repr_str
"""

class PolyOverlaps():
    """2D IoU Calculator"""

    def __call__(self, bboxes1, bboxes2, mode='iou', is_aligned=False):
        if bboxes1.shape[-1] in [5, 6]:
            if bboxes1.shape[-1] == 6:
                bboxes1 = bboxes1[:, :5]
            bboxes1 = bt.bbox2type(bboxes1, 'poly')
        if bboxes2.shape[-1] in [5, 6]:
            if bboxes2.shape[-1] == 6:
                bboxes2 = bboxes1[:, :5]
            bboxes2 = bt.bbox2type(bboxes1, 'poly')

        assert bboxes1.shape[-1] in [0, 8, 9]
        assert bboxes2.shape[-1] in [0, 8, 9]
        if bboxes2.shape[-1] == 9:
            bboxes2 = bboxes2[..., :8]
        if bboxes1.shape[-1] == 9:
            bboxes1 = bboxes1[..., :8]
        with torch.cuda.amp.autocast(enabled=False):
            iou_matrix = bt.bbox_overlaps(bboxes1, bboxes2, mode, is_aligned)
        return iou_matrix

    def __repr__(self):
        """str: a string describing the module"""
        repr_str = self.__class__.__name__ + '()'
        return repr_str


# with torch.cuda.amp.autocast(enabled=False):
# 判断点集是否在目标多边形之内
class GridIsInPoly(Function):
    @staticmethod
    def forward(ctx, x_centers_per_image, y_centers_per_image, gt_bboxes_per_image):
        # fg_mask = torch.full_like(grid_x, False).bool()
        is_in_boxes = torch.zeros(size=x_centers_per_image.shape).bool()
        is_in_boxes = is_in_boxes.to(gt_bboxes_per_image.device)
        x_centers_per_image = x_centers_per_image.cpu().numpy()
        y_centers_per_image = y_centers_per_image.cpu().numpy()
        gt_bboxes_per_image = gt_bboxes_per_image.cpu().numpy()

        for i in range(gt_bboxes_per_image.shape[0]): # num_gt
            target = gt_bboxes_per_image[i]
            target = np.reshape(target, (4, 2))
            for j in range(x_centers_per_image.shape[1]): # num_anchors
                pt = (x_centers_per_image[i,j], y_centers_per_image[i,j])
                is_in_boxes[i, j] = cv2.pointPolygonTest(target, pt, False) > 0
        ctx.mark_non_differentiable(is_in_boxes)
        return is_in_boxes

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        return ()
        # return None, None, None
grid_is_in_poly1 = GridIsInPoly.apply




# 参考
"""
        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_
"""
