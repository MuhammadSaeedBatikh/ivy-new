"""Collection of Mindspore activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional, Union

# global
import mindspore as ms
from mindspore import ops
import mindspore.numpy as msnp
from mindspore._c_expression.typing import Float, Int, Complex
from typing import Tuple, Sequence
from functools import wraps

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

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


def _back_to_original_dtype(dtype=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x_type = args[0].dtype
            if dtype is None:
                ret = func(*args, **kwargs).astype(x_type)
            elif isinstance(dtype, (Tuple, list)):
                if any(t in str(x_type).lower() for t in dtype):
                    ret = func(*args, **kwargs).astype(x_type)
                else:
                    ret = func(*args, **kwargs)
            elif x_type in str(x_type).lower():
                ret = func(*args, **kwargs).astype(x_type)
            else:
                ret = func(*args, **kwargs)
            return ret

        return wrapper

    return decorator


def relu(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return ops.relu(x)


def leaky_relu(
        x: ms.Tensor,
        /,
        *,
        alpha: float = 0.2,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.where(x > 0.0, x, alpha * x)


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float64")}, backend_version)
def gelu(
        x: ms.Tensor,
        /,
        *,
        approximate: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if approximate:
        return (
                0.5 * x * (1 + ops.tanh(((2 / msnp.pi) ** 0.5) * (x + 0.044715 * x ** 3)))
        )
    x = _cast_int_to_float(x)
    sqrt_2 = msnp.sqrt(2.).astype(x.dtype)
    return ms.Tensor(x * (ops.erf(x / sqrt_2) + 1) / 2, dtype=x.dtype)


def sigmoid(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    if not ivy.is_array(x):
        x = ms.Tensor(x)
    return ops.sigmoid(x)


sigmoid.support_native_out = True


def softmax(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[int] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    axis = axis if axis is not None else -1
    return ops.softmax(x, axis)


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float64")}, backend_version)
def softplus(
        x: ms.Tensor,
        /,
        *,
        beta: Optional[Union[int, float]] = 1.,
        threshold: Optional[Union[int, float]] = 20,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x = _cast_int_to_float(x)
    beta = float(beta)
    return ops.log(1 + ops.exp(beta * x)) / beta


@with_unsupported_dtypes({"1.11.0 and below": ("bfloat16", "float64")}, backend_version)
def log_softmax(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[int] = None,
        out: Optional[ms.Tensor] = None,
):
    x = _cast_int_to_float(x)
    axis = axis if axis is not None else -1
    return ops.log_softmax(x, axis)


def mish(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return x * ops.tanh(softplus(x))
