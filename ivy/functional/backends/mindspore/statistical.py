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
from ivy.functional.ivy.statistical import _get_promoted_type_of_operands

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
# -------------------#


def min(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return ops.min(x, axis=axis, keep_dims=keepdims)


min.support_native_out = False


def max(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    axis = tuple(axis) if isinstance(axis, list) else axis
    return ops.max(x, axis=axis, keep_dims=keepdims)


max.support_native_out = False


def mean(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    axis = () if axis is None else axis
    axis = tuple(axis) if isinstance(axis, list) else axis
    return ops.mean(x, axis=axis, keep_dims=keepdims)


mean.support_native_out = False


def _infer_dtype(dtype: Type) -> Type:
    default_dtype = ivy.infer_default_dtype(dtype)
    if default_dtype in ivy.valid_dtypes:
        if ivy.dtype_bits(dtype) < ivy.dtype_bits(default_dtype):
            return ivy.as_native_dtype(default_dtype)
    return ivy.as_native_dtype(dtype)


def prod(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        dtype: Optional[Type] = None,
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        dtype = _infer_dtype(x.dtype)
    axis = tuple(axis) if isinstance(axis, list) else axis
    return ops.prod(x, axis=axis, keep_dims=keepdims)


def std(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        correction: Union[int, float] = 0,
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if axis is None:
        axis = list(range(len(x.shape)))
    if axis == ():
        return x
    axis = (axis,) if isinstance(axis, int) else tuple(axis)
    if correction == 0:
        return ops.std(x, axis=axis, keep_dims=keepdims, unbiased=False)
    elif correction == 1:
        return ops.std(x, axis=axis, keep_dims=keepdims, unbiased=True)
    size = 1
    for a in axis:
        size *= x.shape[a]
    if size - correction <= 0:
        ret = ops.std(x, axis=axis, keep_dims=keepdims, unbiased=False)
        ret = ivy.full(ret.shape, float("nan"), dtype=ret.dtype)
        return ret
    ret = ops.mul(
        ops.std(x, axis=axis, keep_dims=keepdims, unbiased=False),
        (size / (size - correction)) ** 0.5,
    )
    return ret


def sum(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        dtype: Optional[Type] = None,
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if dtype is None and not ivy.is_bool_dtype(x):
        dtype = x.dtype
    axis = tuple(axis) if isinstance(axis, list) else axis
    return msnp.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)


def cumsum(
        x: ms.Tensor,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        *,
        dtype: Optional[Type] = None,
        out: ms.Tensor = None,
) -> ms.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if dtype is ms.bool_:
            dtype = ivy.default_int_dtype()
        elif ivy.is_int_dtype(x.dtype):
            dtype = ivy.promote_types(x.dtype, ivy.default_int_dtype(as_native=True))
        else:
            dtype = _infer_dtype(x.dtype)
        dtype = ivy.as_native_dtype(dtype)
    x = F.cast(x, dtype)
    return ops.CumSum(exclusive=exclusive, reverse=reverse)(x, axis)


def cumprod(
        x: ms.Tensor,
        axis: int = 0,
        exclusive: bool = False,
        reverse: bool = False,
        *,
        dtype: Optional[Type] = None,
        out: ms.Tensor = None,
) -> ms.Tensor:
    dtype = ivy.as_native_dtype(dtype)
    if dtype is None:
        if dtype is ms.bool_:
            dtype = ivy.default_int_dtype()
        elif ivy.is_int_dtype(x.dtype):
            dtype = ivy.promote_types(x.dtype, ivy.default_int_dtype(as_native=True))
        else:
            dtype = _infer_dtype(x.dtype)
        dtype = ivy.as_native_dtype(dtype)
    x = F.cast(x, dtype)
    return ops.CumProd(exclusive=exclusive, reverse=reverse)(x, axis)


def einsum(
        equation: str,
        *operands: Union[ms.Tensor, ms.Parameter],
        out: Optional[Union[ms.Tensor, ms.Parameter]] = None,
) -> Union[ms.Tensor, ms.Parameter]:
    dtype = _get_promoted_type_of_operands(operands)
    operands = tuple(ops.cast(operand, ms.float32) for operand in operands)
    res = ops.Einsum(equation)(operands)
    return ops.cast(res, dtype)
