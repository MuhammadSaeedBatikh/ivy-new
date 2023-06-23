from numbers import Number
from typing import Optional, Tuple, Union
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
import mindspore.ops.functional as F
from mindspore._c_expression.typing import Float, Int, Complex
from mindspore import Type
from typing import List

# local
import ivy
from ivy import promote_types_of_inputs
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


def sinc(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    xdtype = x.dtype
    x = _cast_int_to_float(x)
    y = msnp.pi * msnp.where(x == 0, ms.Tensor(1.0e-20, x.dtype), x)
    res = ops.sin(y) / y
    if 'float' in str(xdtype).lower():
        return res.astype(xdtype)
    else:
        return res


def trapz(
        y: ms.Tensor,
        /,
        *,
        x: Optional[ms.Tensor] = None,
        dx: Optional[float] = 1.0,
        axis: Optional[int] = -1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.trapz(y, x=x, dx=dx, axis=axis)


@with_unsupported_dtypes(
    {"2.0.0 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
)
def float_power(
        x1: Union[ms.Tensor, float, list, tuple],
        x2: Union[ms.Tensor, float, list, tuple],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if ivy.any(ivy.is_complex_dtype(x1)) or ivy.any(ivy.is_complex_dtype(x2)):
        out_dtype = ms.complex128
    else:
        out_dtype = ms.float64
    return msnp.float_power(x1, x2, dtype=out_dtype)


def exp2(
        x: Union[ms.Tensor, float, list, tuple],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.exp2(x)


def angle(
        z: ms.Tensor,
        /,
        *,
        deg: Optional[bool] = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    z = msnp.asarray(z)
    if 'complex' in str(z.dtype).lower():
        zimag = ops.Imag()(z)
        zreal = ops.Real()(z)
    else:
        zimag = 0
        zreal = z
    a = msnp.arctan2(zimag, zreal)
    if deg:
        a *= 180 / msnp.pi
    return a


def imag(
        val: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return ops.Imag()(val)


def hypot(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = _cast_int_to_float(x1), _cast_int_to_float(x2)
    return msnp.hypot(x1, x2)


def real(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return ops.Real()(x)


def conj(
        x: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return ops.Conj()(x)


def frexp(
        x: ms.Tensor,
        /,
        *,
        out: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,
) -> Tuple[ms.Tensor, ms.Tensor]:
    x = _cast_int_to_float(x)
    e = ops.floor(ops.log(ops.abs(x)) / F.cast(ops.log(ms.Tensor(2.0)), x.dtype))
    e = F.cast(e, x.dtype)
    while ops.ReduceAny()(ops.abs(x / ops.pow(2, e)) >= 1):
        e += F.cast(ops.abs(x / ops.pow(2, e)) >= 1, e.dtype)
    m = x / ops.pow(2, e)
    e = F.cast(e, ms.int32)
    return m, e


def lcm(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return msnp.abs(
        msnp.lcm(
            x1,
            x2)
    )


def ldexp(
        x1: ms.Tensor,
        x2: Union[ms.Tensor, int, list, tuple],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return ops.ldexp(x1, x2)


def xlogy(
        x: ms.Tensor, y: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x, y = promote_types_of_inputs(x, y)
    if (x == 0).all():
        return 0.0
    else:
        return x * msnp.log(y)


def nextafter(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = _cast_int_to_float(x1), _cast_int_to_float(x2)
    return ops.NextAfter()(x1, x2)


def fmod(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return msnp.fmod(
        x1,
        x2,
    )


def fix(
        x: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.fix(x)


def signbit(
        x: Union[ms.Tensor, float, int, list, tuple],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.signbit(x)


def logaddexp2(
        x1: Union[ms.Tensor, int, list, tuple],
        x2: Union[ms.Tensor, int, list, tuple],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    if not ivy.is_float_dtype(x1):
        x1 = x1.astype(ivy.default_float_dtype(as_native=True))
        x2 = x2.astype(ivy.default_float_dtype(as_native=True))
    return msnp.logaddexp2(x1, x2)


def gcd(
        x1: Union[ms.Tensor, int, list, tuple],
        x2: Union[ms.Tensor, float, list, tuple],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = promote_types_of_inputs(x1, x2)
    return msnp.gcd(x1, x2)


def count_nonzero(
        a: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: Optional[bool] = False,
        dtype: Optional[ms.Type] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if isinstance(axis, list):
        axis = tuple(axis)
    ret = msnp.count_nonzero(a, axis=axis, keepdims=keepdims)
    if msnp.isscalar(ret):
        return ms.Tensor(ret, dtype)
    return ret.astype(dtype)


def nansum(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[Tuple[int, ...], int]] = None,
        dtype: Optional[ms.Type] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if isinstance(axis, list):
        axis = tuple(axis)
    return msnp.nansum(x, axis=axis, dtype=dtype, keepdims=keepdims)


def isclose(
        a: ms.Tensor,
        b: ms.Tensor,
        /,
        *,
        rtol: Optional[float] = 1e-05,
        atol: Optional[float] = 1e-08,
        equal_nan: Optional[bool] = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    ret = msnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if msnp.isscalar(ret):
        return ms.Tensor(ret, ms.bool_)
    return ret


def allclose(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        rtol: Optional[float] = 1e-05,
        atol: Optional[float] = 1e-08,
        equal_nan: Optional[bool] = False,
        out: Optional[ms.Tensor] = None,
) -> bool:
    res = ops.all(msnp.isclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan))
    return bool(res)


def diff(
        x: Union[ms.Tensor, list, tuple],
        /,
        *,
        n: int = 1,
        axis: int = -1,
        prepend: Optional[Union[ms.Tensor, int, float, list, tuple]] = None,
        append: Optional[Union[ms.Tensor, int, float, list, tuple]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    prepend = prepend if prepend is not None else None
    append = append if append is not None else None
    return msnp.diff(x, n=n, axis=axis, prepend=prepend, append=append)


def gradient(
        x: ms.Tensor,
        /,
        *,
        spacing: Optional[Union[int, list, tuple]] = 1,
        axis: Optional[Union[int, list, tuple]] = None,
        edge_order: Optional[int] = 1,
) -> Union[ms.Tensor, List[ms.Tensor]]:
    if type(spacing) in (int, float):
        return msnp.gradient(x, spacing, axis=axis, edge_order=edge_order)
    return msnp.gradient(x, *spacing, axis=axis, edge_order=edge_order)

