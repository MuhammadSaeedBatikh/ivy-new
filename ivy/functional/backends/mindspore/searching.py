# global

from numbers import Number
from typing import Optional, Tuple, Union
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore._c_expression.typing import Float, Int, Complex
from mindspore import Type

# local
import ivy

# Redundant Util ... to be moved
int_to_float_dict = {
    str(ms.int8): ms.float16,
    str(ms.int16): ms.float16,
    str(ms.int32): ms.float32,
    str(ms.int64): ms.float64,
}


def _cast_int_to_float(x: ms.Tensor) -> ms.Tensor:
    if isinstance(x.dtype, Float) or isinstance(x.dtype, Complex):
        return x

    elif isinstance(x.dtype, Int):
        x_type = str(x.dtype)
        new_type = int_to_float_dict[x_type]
        return x.astype(new_type)
    else:
        raise TypeError(f'Unsupported Type:{x.dtype}')


# Array API Standard #
# ------------------ #

def argmin(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[int] = None,
        keepdims: bool = False,
        output_dtype: Optional[Type] = None,
        select_last_index: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if select_last_index:
        if axis is None:
            x = msnp.flip(x, axis=([axes for axes in range(x.ndim)]))
            ret = ops.argmin(x, axis, keepdims)
            ret = x.numel() - ret - 1
        else:
            x = msnp.flip(x, axis=(axis,))
            ret = ops.argmin(x, axis, keepdims)
            ret = x.shape[axis] - ret - 1
    else:
        ret = ops.argmin(x, axis, keepdims)
    return ret


def argmax(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[int] = None,
        keepdims: bool = False,
        output_dtype: Optional[Type] = None,
        select_last_index: bool = False,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if select_last_index:
        if axis is None:
            x = msnp.flip(x, axis=([axes for axes in range(x.ndim)]))
            ret = ops.argmax(x, axis, keepdims)
            ret = x.numel() - ret - 1
        else:
            x = msnp.flip(x, axis=(axis,))
            ret = ops.argmax(x, axis, keepdims)
            ret = x.shape[axis] - ret - 1
    else:
        ret = ops.argmax(x, axis, keepdims)
    if dtype:
        dtype = ivy.as_native_dtype(dtype)
        return ret.to(dtype=dtype)
    return ret


def nonzero(
        x: ms.Tensor,
        /,
        *,
        as_tuple: bool = True,
        size: Optional[int] = None,
        fill_value: Number = 0,
) -> Union[ms.Tensor, Tuple[ms.Tensor]]:
    res = ops.nonzero(x)
    if size is not None:
        if isinstance(fill_value, float):
            res = msnp.asarray(res, dtype=ms.float64)

        diff = size - res[0].shape[0]
        if diff > 0:
            res = msnp.pad(res, ((0, 0), (0, diff)), constant_values=fill_value)
        elif diff < 0:
            res = msnp.array(res)[:, :size]
    if as_tuple:
        return tuple(res)
    return msnp.stack(res, axis=1)


def where(
    condition: ms.Tensor,
    x1: ms.Tensor,
    x2: ms.Tensor,
    /,
    *,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ivy.astype(msnp.where(condition, x1, x2), x1.dtype, copy=False)


def argwhere(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    # TODO: transpose does not handle tensors with axis with 0 elements.
    if x.shape == (1,):
        return ms.Tensor([[0]], dtype=ms.int64) if x[0] else ms.Tensor([], dtype=ms.int64)
    else:
        return msnp.transpose(ops.nonzero(x))