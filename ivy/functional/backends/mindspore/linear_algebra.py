# global

import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore._c_expression.typing import Float, Int, Complex, UInt
from typing import Union, Optional, Tuple, Literal, List, NamedTuple, Sequence

from collections import namedtuple

# local
import ivy
from ivy import inf
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

# Helpers


int_to_float_dict = {
    str(ms.uint8): ms.float16,
    str(ms.uint16): ms.float16,
    str(ms.uint32): ms.float32,
    str(ms.uint64): ms.float32,
    str(ms.int8): ms.float16,
    str(ms.int16): ms.float16,
    str(ms.int32): ms.float32,
    str(ms.int64): ms.float32,
}


def _cast_int_to_float(x: ms.Tensor) -> ms.Tensor:
    if isinstance(x.dtype, Float) or isinstance(x.dtype, Complex):
        return x

    elif isinstance(x.dtype, (Int, UInt)):
        x_type = str(x.dtype)
        new_type = int_to_float_dict[x_type]
        return x.astype(new_type)
    else:
        raise TypeError(f'Unsupported Type:{x.dtype}')


# Array API Standard #
# -------------------#


def cholesky(
        x: ms.Tensor, /, *, upper: bool = False, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    ret = ops.cholesky(x, upper=upper)
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


cholesky.support_native_out = True


def cross(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        axisa: int = -1,
        axisb: int = -1,
        axisc: int = -1,
        axis: int = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.cross(a=x1, b=x2, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


cross.support_native_out = True


def _det_2x2(a):
    return (a[..., 0, 0] * a[..., 1, 1] -
            a[..., 0, 1] * a[..., 1, 0])


def _det_3x3(a):
    return (a[..., 0, 0] * a[..., 1, 1] * a[..., 2, 2] +
            a[..., 0, 1] * a[..., 1, 2] * a[..., 2, 0] +
            a[..., 0, 2] * a[..., 1, 0] * a[..., 2, 1] -
            a[..., 0, 2] * a[..., 1, 1] * a[..., 2, 0] -
            a[..., 0, 0] * a[..., 1, 2] * a[..., 2, 1] -
            a[..., 0, 1] * a[..., 1, 0] * a[..., 2, 2])


def det(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    if x.shape[-2] == 2:
        return _det_2x2(x)
    if x.shape[-2] == 3:
        return _det_3x3(x)
    return _cast_int_to_float(x).matrix_determinant()


det.support_native_out = True


def diagonal(
        x: ms.Tensor,
        /,
        *,
        offset: int = 0,
        axis1: int = -2,
        axis2: int = -1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def inner(
        x1: ms.Tensor, x2: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return msnp.inner(x1, x2)


inner.support_native_out = True


def inv(
        x: ms.Tensor,
        /,
        *,
        adjoint: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    # costly check! + why return same tensor if singular?!
    if ops.any(det(x) == 0):
        return x
    else:
        x = _cast_int_to_float(x)
        if not adjoint:
            ret = ops.MatrixInverse()(x)
            return ret
        else:
            x = ops.adjoint(x)
            ret = ops.MatrixInverse()(x)
            return ret


inv.support_native_out = True


def matmul(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        transpose_a: bool = False,
        transpose_b: bool = False,
        adjoint_a: bool = False,
        adjoint_b: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if transpose_a:
        x1 = x1.transpose()
    if transpose_b:
        x2 = x2.transpose()
    if adjoint_a:
        x1 = ops.adjoint(x1)
    if adjoint_b:
        x2 = ops.adjoint(x2)
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.matmul(_cast_int_to_float(x1), _cast_int_to_float(x2)).astype(x1.dtype)


matmul.support_native_out = True


@with_unsupported_dtypes({"2.0.0 and below": ("float16", "bfloat16")}, backend_version)
def matrix_norm(
        x: ms.Tensor,
        /,
        *,
        ord: Optional[Union[int, float, Literal[inf, -inf, "fro", "nuc"]]] = "fro",
        axis: Optional[Tuple[int, int]] = (-2, -1),
        keepdims: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x = _cast_int_to_float(x)
    if ord == 'nuc':
        return ops.NuclearNorm(dim=axis, keepdim=keepdims)(x)
    elif ord in ['inf', '-inf']:
        ord = float(ord)
    return msnp.norm(x, ord=ord, axis=axis, keepdims=keepdims)


matrix_norm.support_native_out = True


def matrix_power(
        x: ms.Tensor, n: int, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    return msnp.matrix_power(x, n)


matrix_power.support_native_out = True


def matrix_transpose(
        x: ms.Tensor, /, *, conjugate: bool = False, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    if conjugate:
        ops.conj(x)
    return msnp.swapaxes(x, -1, -2)


def outer(
        x1: ms.Tensor, x2: ms.Tensor, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return ops.outer(x1, x2)


outer.support_native_out = True


@with_unsupported_dtypes({"1.11.0 and below": ("float16", "bfloat16")}, backend_version)
def pinv(
        x: ms.Tensor,
        /,
        *,
        rtol: Optional[Union[float, Tuple[float]]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x = _cast_int_to_float(x)
    return ops.pinv(x, rtol=rtol)


pinv.support_native_out = True


def eig(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> Tuple[ms.Tensor]:
    result_tuple = NamedTuple(
        "eig", [("eigenvalues", ms.Tensor), ("eigenvectors", ms.Tensor)]
    )
    # unordered, complex eigenvalues. Check if res.conj() == res: ops.Real()(res)
    eigenvalues, eigenvectors = ops.Eig(True)(_cast_int_to_float(x))
    return result_tuple(eigenvalues, eigenvectors)


def trace(
        x: ms.Tensor,
        /,
        *,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.trace(x, offset=offset, axis1=axis1, axis2=axis2)


# Extra #
# ----- #


def diag(
        x: ms.Tensor,
        /,
        *,
        k: int = 0,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.diag(x, k=k)
