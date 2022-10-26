# coding: utf-8

import torch
from torch.autograd import Function
if torch.cuda.is_available():
    from . import convex_cuda as convex_ext
else:
    from . import convex_cpu as convex_ext


#  with torch.cuda.amp.autocast(enabled=False):
class ConvexSortFunction(Function):

    @staticmethod
    def forward(ctx, pts, masks, circular):
        idx = convex_ext.convex_sort(pts, masks, circular)
        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        return ()


def convex_sort(pts, masks, circular=True):
    return ConvexSortFunction.apply(pts, masks, circular)


