from typing import Optional, Union

# global
import mindspore as ms
import mindspore.numpy as msnp
import mindspore.ops as ops


# local


def logit(x: ms.Tensor, /, *, eps: Optional[float] = None, out=None):
    return ops.logit(x, eps=eps)


def thresholded_relu(
        x: ms.Tensor,
        /,
        *,
        threshold: Optional[Union[int, float]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.where(x > threshold, x, 0)


def logsigmoid(input: ms.Tensor) -> ms.Tensor:
    return ops.Softplus()(-input)


def relu6(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return ops.relu6(x)


def selu(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return ops.selu(x)
