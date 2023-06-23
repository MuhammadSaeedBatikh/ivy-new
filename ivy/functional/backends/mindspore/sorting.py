# global
from mindspore._c_expression.typing import Float, Int, Complex
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
from typing import Optional

# local
import ivy


def argsort(
        x: ms.Tensor,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    # does not support stable
    x = x.astype(ms.float32) if x.dtype == ms.float64 else x
    _, sorted_indices = ops.Sort(axis=axis, descending=descending)(x)
    return sorted_indices


def sort(
        x: ms.Tensor,
        /,
        *,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    # does not support stable
    x = x.astype(ms.float32) if x.dtype == ms.float64 else x
    sorted_tensor, _ = ops.Sort(axis=axis, descending=descending)(x)
    return sorted_tensor


def searchsorted(
    x: ms.Tensor,
    v: ms.Tensor,
    /,
    *,
    side="left",
    sorter=None,
    ret_dtype=ms.int64,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    assert ivy.is_int_dtype(ret_dtype), ValueError(
        "only Integer data types are supported for ret_dtype."
    )
    is_sorter_provided = sorter is not None
    if is_sorter_provided:
        assert ivy.is_int_dtype(sorter.dtype) and not ivy.is_uint_dtype(
            sorter.dtype
        ), TypeError(
            f"Only signed integer data type for sorter is allowed, got {sorter.dtype}."
        )
    if x.ndim != 1:
        assert x.shape[:-1] == v.shape[:-1], RuntimeError(
            f"the first N-1 dimensions of x array and v array "
            f"must match, got {x.shape} and {v.shape}"
        )
        if is_sorter_provided:
            x = msnp.take_along_axis(x, sorter, axis=-1)
        original_shape = v.shape
        x = x.reshape(-1, x.shape[-1])
        v = v.reshape(-1, v.shape[-1])
        out_array = []
        for i in range(x.shape[0]):
            out_array += [msnp.searchsorted(x[i], v[i], side=side).numpy()]
        out_array = ms.Tensor(out_array)
        ret = out_array.reshape(original_shape)
    else:
        ret = msnp.searchsorted(x, v, side=side, sorter=sorter)
    return ret.astype(ret_dtype)
