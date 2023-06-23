# global
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp

# local
from typing import Union, Optional, Sequence
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

@with_unsupported_dtypes({"2.0.0 and below": ("complex64", "complex128")}, backend_version)
def all(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x = x.astype(dtype=ms.bool_)
    if axis is None:
        num_dims = len(x.shape)
        axis = list(range(num_dims))
    if isinstance(axis, int):
        return ops.all(x, axis=axis, keep_dims=keepdims)
    dims = len(x.shape)
    axis = [i % dims for i in axis]
    axis.sort()
    for i, a in enumerate(axis):
        x = ops.all(x, axis=a if keepdims else a - i, keep_dims=keepdims)
    return x


@with_unsupported_dtypes({"2.0.0 and below": ("complex64", "complex128")}, backend_version)
def any(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    axis = () if axis is None else axis
    return ops.any(x.astype(ms.bool_), axis=axis, keep_dims=keepdims)


any.support_native_out = False
