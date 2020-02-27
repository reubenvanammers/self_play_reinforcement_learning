import warnings

import torch
import torch.nn._reduction as _Reduction


def _weighted_smooth_l1_loss(input, target, weights):
    # type: (Tensor, Tensor) -> Tensor
    t = torch.abs(input - target)
    return weights * torch.where(t < 1, 0.5 * t ** 2, t - 0.5)


def weighted_smooth_l1_loss(input, target, weights, size_average=None, reduce=None, reduction="mean"):
    # type: (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """
    if not (target.size() == input.size()):
        warnings.warn(
            "Using a target size ({}) that is different to the input size ({}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.".format(target.size(), input.size()),
            stacklevel=2,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = _weighted_smooth_l1_loss(input, target, weights)
        if reduction != "none":
            ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)
    else:
        raise (ValueError("haven't thought this through"))
        expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        ret = torch._C._nn.smooth_l1_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret
