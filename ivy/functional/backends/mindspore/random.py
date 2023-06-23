# global

from typing import Union, Optional, Sequence
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore.ops import functional as F
from mindspore._c_expression.typing import Float, Int, Complex
from mindspore import Type

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from ivy.functional.ivy.random import (
    _check_bounds_and_get_shape,
    _randint_check_dtype_and_bound,
    _check_valid_scale,
)

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
@with_unsupported_dtypes({"2.0.0 and below": ("float64")}, backend_version)
def random_uniform(
        *,
        low: Union[float, ms.Tensor] = 0.0,
        high: Union[float, ms.Tensor] = 1.0,
        shape: Optional[Union[ivy.NativeShape, Sequence[int], ms.Tensor]] = None,
        dtype: Type,
        device: str,
        out: Optional[ms.Tensor] = None,
        seed: Optional[int] = None,
) -> ms.Tensor:
    shape = _check_bounds_and_get_shape(low, high, shape)
    if dtype is None:
        return ops.uniform(shape, low, high, seed=seed)
    return ops.uniform(shape, low, high, seed=seed).astype(dtype)

@with_unsupported_dtypes({"2.0.0 and below": ("float64")}, backend_version)
def random_normal(
        *,
        mean: Union[float, ms.Tensor] = 0.0,
        std: Union[float, ms.Tensor] = 1.0,
        shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
        device: str,
        dtype: Type = None,
        seed: Optional[int] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    _check_valid_scale(std)
    shape = _check_bounds_and_get_shape(mean, std, shape)
    if dtype is None:
        return ops.normal(shape, mean.astype(ms.float32), std.astype(ms.float32), seed=seed)
    else:
        return ops.normal(shape, mean.astype(ms.float32), std.astype(ms.float32), seed=seed).astype(dtype)

@with_unsupported_dtypes({"2.0.0 and below": ("float64")}, backend_version)
def multinomial(
        population_size: int,
        num_samples: int,
        /,
        *,
        batch_size: int = 1,
        probs: Optional[ms.Tensor] = None,
        replace: bool = True,
        device: str,
        seed: Optional[int] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if probs is None:
        probs = (ops.ones((batch_size, population_size,)) / population_size)
    return ops.multinomial(probs, num_samples, replace, seed=seed)

@with_unsupported_dtypes({"2.0.0 and below": ("float64")}, backend_version)
def randint(
        low: Union[float, ms.Tensor],
        high: Union[float, ms.Tensor],
        /,
        *,
        shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
        device: str,
        dtype: Optional[Union[Type, ivy.Dtype]] = None,
        seed: Optional[int] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if not dtype:
        dtype = ivy.default_int_dtype()
    dtype = ivy.as_native_dtype(dtype)
    _randint_check_dtype_and_bound(low, high, dtype)
    shape = _check_bounds_and_get_shape(low, high, shape)
    low = _cast_int_to_float(low)
    high = _cast_int_to_float(high)
    if dtype is None:
        return ops.uniform(shape, low, high, seed=seed)
    return ops.uniform(shape, low, high, seed=seed).astype(dtype)


def seed(*, seed_value: int = 0) -> None:
    ms.set_seed(seed_value)


def shuffle(
        x: ms.Tensor,
        /,
        *,
        seed: Optional[int] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return ops.shuffle(x, seed=seed)
