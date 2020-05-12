from torch import nn
import functools


class FlattenedLoss:
    "Same as `func`, but flattens input and target."

    def __init__(
        self, func, *args, axis: int = -1, floatify: bool = False, is_2d: bool = True, **kwargs,
    ):
        self.func, self.axis, self.floatify, self.is_2d = (
            func(*args, **kwargs),
            axis,
            floatify,
            is_2d,
        )
        functools.update_wrapper(self, self.func)

    def __repr__(self):
        return f"FlattenedLoss of {self.func}"

    @property
    def reduction(self):
        return self.func.reduction

    @reduction.setter
    def reduction(self, v):
        self.func.reduction = v

    @property
    def weight(self):
        return self.func.weight

    @weight.setter
    def weight(self, v):
        self.func.weight = v

    def __call__(self, input, target, **kwargs):
        input = input.transpose(self.axis, -1).contiguous()
        target = target.transpose(self.axis, -1).contiguous()
        if self.floatify:
            target = target.float()
        input = input.view(-1, input.shape[-1]) if self.is_2d else input.view(-1)
        return self.func.__call__(input, target.view(-1), **kwargs)


def MSELossFlat(*args, axis: int = -1, floatify: bool = True, **kwargs):
    "Same as `nn.MSELoss`, but flattens input and target."
    return FlattenedLoss(nn.MSELoss, *args, axis=axis, floatify=floatify, is_2d=False, **kwargs)
