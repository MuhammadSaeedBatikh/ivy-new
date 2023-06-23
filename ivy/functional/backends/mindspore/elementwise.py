# global
from typing import Union, Optional
from functools import wraps

import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore._c_expression.typing import Float, Int, Complex

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from . import backend_version


def _back_to_original_dtype(dtype=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x_type = args[0].dtype
            if dtype is None:
                ret = func(*args, **kwargs).astype(x_type)
            elif dtype in str(x_type).lower():
                ret = func(*args, **kwargs).astype(x_type)
            else:
                ret = func(*args, **kwargs)
            return ret

        return wrapper

    return decorator


def _cast_for_unary_op(x):
    if not isinstance(x, ms.Tensor):
        x = ms.Tensor(x)
    return x


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


def add(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        alpha: Optional[Union[int, float]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        x2 = ops.multiply(x2, alpha)
    return ops.add(x1, x2)


add.support_native_out = True


def bitwise_xor(
        x1: Union[int, bool, ms.Tensor],
        x2: Union[int, bool, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.bitwise_xor(x1, x2)


bitwise_xor.support_native_out = True


def expm1(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.expm1(x)


expm1.support_native_out = True


def bitwise_invert(
        x: Union[int, bool, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.invert(x)


bitwise_invert.support_native_out = True


def isfinite(x: ops.Tensor, /, *, out: Optional[ops.Tensor] = None) -> ops.Tensor:
    x = _cast_for_unary_op(x)
    return ops.isfinite(x)


def isinf(
        x: ms.Tensor,
        /,
        *,
        detect_positive: bool = True,
        detect_negative: bool = True,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if 'float' not in str(x.dtype).lower():
        x = x.astype(ms.float32)
    if detect_negative and detect_positive:
        return msnp.isinf(x)
    elif detect_negative:
        return msnp.isneginf(x)
    elif detect_positive:
        return msnp.isposinf(x)
    return msnp.full_like(x, False, dtype=ms.bool_)


def equal(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.equal(x1, x2)


equal.support_native_out = False


def less_equal(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.less_equal(x1, x2)


less_equal.support_native_out = True


def bitwise_and(
        x1: Union[int, bool, ms.Tensor],
        x2: Union[int, bool, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.bitwise_and(x1, x2)


bitwise_and.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float64",)}, backend_version)
def ceil(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype).lower():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return ops.ceil(x)


ceil.support_native_out = True


def floor(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype).lower():
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return ops.floor(x)


floor.support_native_out = True


@_back_to_original_dtype()
def asin(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.asin(x)


asin.support_native_out = False


def asinh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.asinh(x)


asinh.support_native_out = False


def acos(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.acos(x)


def acosh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.acosh(x)


def sign(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    y = _cast_for_unary_op(x)
    if 'uint' in str(x.dtype).lower():
        y = y.astype(ms.int64)
    return ops.Sign()(y).astype(x.dtype)


sign.support_native_out = True


def sqrt(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.sqrt(x)


sqrt.support_native_out = True


def cosh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.cosh(x)


cosh.support_native_out = True


def negative(
        x: Union[float, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.neg(x)


@_scalar_output_to_0d_array
def not_equal(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return msnp.not_equal(x1, x2)


def log10(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.log10(x)


log10.support_native_out = True


def log2(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.log2(x)


def log1p(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.log1p(x)


log1p.support_native_out = True


def isnan(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.isnan(x)


@_scalar_output_to_0d_array
def less(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.less(x1, x2, )


less.support_native_out = True


def multiply(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.multiply(x1, x2)


multiply.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def cos(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.cos(x)


cos.support_native_out = True


def logical_not(
        x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.logical_not(x.astype(ms.bool_))


logical_not.support_native_out = True


def bitwise_or(
        x1: Union[int, bool, ms.Tensor],
        x2: Union[int, bool, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.bitwise_or(x1, x2)


def bitwise_xor(
        x1: Union[int, bool, ms.Tensor],
        x2: Union[int, bool, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.bitwise_xor(x1, x2)


bitwise_or.support_native_out = True


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def remainder(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        modulus: bool = True,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = msnp.where(res >= 0, msnp.floor(res), msnp.ceil(res))
        diff = msnp.asarray(res - res_floored, dtype=res.dtype)
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        return msnp.asarray(msnp.round(diff * x2), dtype=x1.dtype)
    return msnp.remainder(x1, x2)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def sinh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = x.dtype
    if 'float' not in str(x_type).lower() and 'complex' not in str(x_type).lower():
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.sinh(x).astype(x_type)


sinh.support_native_out = True


@_back_to_original_dtype('int8')
@_scalar_output_to_0d_array
def minimum(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        use_where: bool = True,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if x1.dtype is ms.int8:
        x1, x2 = _cast_int_to_float(x1), _cast_int_to_float(x2)
    if use_where:
        ret = msnp.where(x1 <= x2, x1, x2)
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret
    return msnp.minimum(x1, x2)


@_back_to_original_dtype('int8')
@_scalar_output_to_0d_array
def maximum(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        use_where: bool = True,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if x1.dtype is ms.int8:
        x1, x2 = _cast_int_to_float(x1), _cast_int_to_float(x2)
    if use_where:
        ret = msnp.where(x1 <= x2, x1, x2)
        if ivy.exists(out):
            return ivy.inplace_update(out, ret)
        return ret
    return msnp.maximum(x1, x2)


def positive(
        x: Union[float, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.positive(x)


def square(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = x.dtype
    if 'float' not in str(x_type).lower():
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.square(x).astype(x_type)


square.support_native_out = True


def pow(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.pow(x1, x2)


pow.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "complex")}, backend_version)
def round(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return ops.round(x)


round.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("complex",)}, backend_version)
def trunc(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if "int" not in str(x.dtype).lower():
        ret = ops.trunc(x)
    else:
        ret = x
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("int8", "complex")}, backend_version)
def abs(
        x: Union[float, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    if "uint" in ivy.dtype(x):
        return x
    return ops.abs(x)


abs.support_native_out = False


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "complex")}, backend_version)
def logaddexp(
        x1: ms.Tensor, x2: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x_type = str(x1.dtype).lower()
    if 'float' not in x_type:
        x1 = x1.astype(ms.float32)
    x_type = str(x2.dtype).lower()
    if 'float' not in x_type:
        x2 = x2.astype(ms.float32)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.logaddexp(x1, x2)


logaddexp.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def tan(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.tan(x)


tan.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def tanh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.tanh(x)


def atan(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.atan(x)


atan.support_native_out = True


def atan2(
        x1: ms.Tensor, x2: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x_type = str(x1.dtype).lower()
    if 'float' not in x_type:
        x1 = x1.astype(ms.float32)
    x_type = str(x2.dtype).lower()
    if 'float' not in x_type:
        x2 = x2.astype(ms.float32)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.atan2(x1, x2)


atan2.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def log(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.log(x)


log.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def exp(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)
    x = _cast_for_unary_op(x)
    return ops.exp(x)


exp.support_native_out = True


def subtract(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        alpha: Optional[Union[int, float]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if alpha not in (1, None):
        x2 = ops.multiply(x2, alpha)
    return msnp.subtract(x1, x2)


subtract.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def atanh(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type and 'complex' not in x_type:
        x = x.astype(ms.float32)

    x = _cast_for_unary_op(x)
    return ops.atanh(x)


atanh.support_native_out = True


# Extra #
# ------#


@with_unsupported_dtypes({"1.11.0 and below": ("float64", "complex")}, backend_version)
def erf(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)

    x = _cast_for_unary_op(x)
    return ops.erf(x)


erf.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, backend_version)
def reciprocal(
        x: Union[float, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return ops.Reciprocal()(x)


reciprocal.support_native_out = True


def deg2rad(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)

    return ops.deg2rad(x)


deg2rad.support_native_out = True


def rad2deg(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    x_type = str(x.dtype).lower()
    if 'float' not in x_type:
        x = x.astype(ms.float32)

    return ops.rad2deg(x)


rad2deg.support_native_out = True


def isreal(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return ops.isreal(x)


@with_unsupported_dtypes({"1.11.0 and below": ("float", "complex")}, backend_version)
@_scalar_output_to_0d_array
def bitwise_right_shift(
        x1: Union[int, bool, ms.Tensor],
        x2: Union[int, bool, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.RightShift()(x1, x2)


@with_unsupported_dtypes({"1.11.0 and below": ("float", "complex")}, backend_version)
@_scalar_output_to_0d_array
def bitwise_left_shift(
        x1: Union[int, bool, ms.Tensor],
        x2: Union[int, bool, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2, array_api_promotion=True)
    return ops.LeftShift()(x1, x2)


@with_unsupported_dtypes({"1.11.0 and below": ("float", "complex")}, backend_version)
@_scalar_output_to_0d_array
def bitwise_invert(
        x: Union[int, bool, ms.Tensor], /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x = _cast_for_unary_op(x)
    return msnp.invert(x)


@with_unsupported_dtypes({"1.11.0 and below": ("complex")}, backend_version)
@_scalar_output_to_0d_array
def greater(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return msnp.greater(x1, x2)


@with_unsupported_dtypes({"1.11.0 and below": ("complex")}, backend_version)
@_scalar_output_to_0d_array
def greater_equal(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return msnp.greater_equal(x1, x2)


@with_unsupported_dtypes({"1.11.0 and below": ("complex")}, backend_version)
@_scalar_output_to_0d_array
def divide(
        x1: Union[float, ms.Tensor],
        x2: Union[float, ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = msnp.divide(x1, x2)
    if ivy.is_float_dtype(x1):
        ret = msnp.asarray(ret, dtype=x1.dtype)
    else:
        ret = msnp.asarray(ret, dtype=ivy.default_float_dtype(as_native=True))
    return ret
