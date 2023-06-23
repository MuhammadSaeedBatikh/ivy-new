from numbers import Number
from typing import Optional, Tuple, Union
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
import mindspore.ops.functional as F
from mindspore._c_expression.typing import Float, Int, Complex
from mindspore import Type

# local
import ivy
from ivy import promote_types_of_inputs

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


def l2_normalize(x: ms.Tensor, /, *, axis: int = None, out=None) -> ms.Tensor:
    xdtype = x.dtype
    x = x.astype(ms.float32) if isinstance(xdtype, ms.Int) or xdtype == ms.float64 else x
    if axis is None:
        denorm = msnp.norm(x.flatten(), axis=axis, ord=2)
    else:
        denorm = msnp.norm(x, axis=axis, ord=2, keepdims=True)
    denorm = ops.maximum(denorm, 1e-12)
    return (x / denorm).astype(xdtype)


def lp_normalize(
        x: ms.Tensor, /, *, p: float = 2, axis: int = None, out=None
) -> ms.Tensor:
    xdtype = x.dtype
    x = x.astype(ms.float32) if isinstance(xdtype, ms.Int) or xdtype == ms.float64 else x
    if axis is None:
        denorm = msnp.norm(x.flatten(), axis=axis, ord=p)
    else:
        denorm = msnp.norm(x, axis=axis, ord=p, keepdims=True)
    denorm = ops.maximum(denorm, 1e-12)
    return msnp.divide(x, denorm).astype(xdtype)
