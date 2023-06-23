from typing import Optional, Tuple, Union, Sequence
from functools import reduce

# global
import mindspore as ms
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.numpy as msnp
import numpy as np
from mindspore._c_expression.typing import Float, Int, Complex

# local
import ivy
from ivy.functional.ivy.experimental.linear_algebra import _check_valid_dimension_size
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


def _flatten_along_axes(input, order='C', *, start_dim=1, end_dim=-1):
    def check_axis_valid(axis, ndim):
        if axis < -ndim or axis >= ndim:
            raise ValueError("'start_dim' or 'end_dim' out of range.")

    def check_dim_valid(start_dim, end_dim):
        if start_dim > end_dim:
            raise ValueError("For 'flatten', 'start_dim' cannot come after 'end_dim'.")

    def canonicalize_axis(axis, x_rank):
        ndim = x_rank if x_rank != 0 else 1
        check_axis_valid(axis, ndim)
        return axis if axis >= 0 else axis + ndim

    # Check the types of arguments.
    if not isinstance(input, ms.Tensor):
        raise TypeError(f"For 'flatten', argument 'input' must be Tensor.")
    if not isinstance(start_dim, int) or not isinstance(end_dim, int) or \
            isinstance(start_dim, bool) or isinstance(end_dim, bool):
        raise TypeError(f"For 'flatten', both 'start_dim' and 'end_dim' must be int.")
    if order == 'F':
        perm = ops.make_range(0, ops.rank(input))
        new_order = ops.tuple_reversed(perm)
        input = ops._get_cache_prim(P.Transpose)()(input, new_order)

    # Handle the default case.
    x_shape = input.shape
    x_rank = ops.rank(input)
    if start_dim == 1 and end_dim == -1:
        if x_rank in (0, 1):
            return ops.reshape(input, (-1,))
        return ops._get_cache_prim(P.Flatten)()(input)

    # Check axis.
    start_dim = canonicalize_axis(start_dim, x_rank)
    end_dim = canonicalize_axis(end_dim, x_rank)
    check_dim_valid(start_dim, end_dim)
    # If input is a 0-dimensional Tensor, a 1-dimensional Tensor will be returned.
    if x_rank in (0, 1):
        return ops.reshape(input, (-1,))
    # If no dimensions to flatten, return the original object.
    if start_dim == end_dim:
        return input
    # Flatten elements along specified dimensions.
    dim_length = 1
    idx = start_dim
    while idx <= end_dim:
        dim_length *= x_shape[idx]
        idx += 1
    new_shape = x_shape[:start_dim] + (dim_length,) + x_shape[end_dim + 1:]
    return ops.reshape(input, new_shape)


# This is an adopotion of https://github.com/Lezcano/expm Pytorch implementation of expm_taylor to Mindspore
# It is an implementation of Computing the Matrix Exponential with an Optimized Taylor Polynomial Approximation.
# https://www.mdpi.com/2227-7390/7/12/1174

def _matrix_power_two_batch(A, k):
    orig_size = A.shape
    A, k = _flatten_along_axes(A, start_dim=0, end_dim=-3), k.flatten().astype(ms.int64)
    ksorted, idx = ops.sort(k)
    count = msnp.bincount(ksorted)
    nonzero = ops.nonzero(count)
    A = msnp.matrix_power(A, 2 ** int(ksorted[0]))
    last = ksorted[0]
    processed = count[nonzero[0]]
    for exp in nonzero[1:]:
        new, last = exp - last, exp
        A[idx[processed:]] = msnp.matrix_power(A[idx[processed:]], 2 ** new.item())
        processed += count[exp]
    return A.reshape(orig_size)


thetas_dict = {"single": [1.192092800768788e-07,  # m_vals = 1
                          5.978858893805233e-04,  # m_vals = 2
                          5.116619363445086e-02,  # m_vals = 4
                          5.800524627688768e-01,  # m_vals = 8
                          1.461661507209034e+00,  # m_vals = 12
                          3.010066362817634e+00],  # m_vals = 18
               "double": [
                   2.220446049250313e-16,  # m_vals = 1
                   2.580956802971767e-08,  # m_vals = 2
                   3.397168839976962e-04,  # m_vals = 4
                   4.991228871115323e-02,  # m_vals = 8
                   2.996158913811580e-01,  # m_vals = 12
                   1.090863719290036e+00]  # m_vals = 18
               }
degs = [1, 2, 4, 8, 12, 18]


def _taylor_approx(A: ms.Tensor, deg):
    batched = A.ndimension() > 2
    I = ops.eye(A.shape[-2], A.shape[-1], A.dtype)
    if batched:
        I = I.expand_as(A)
    if deg >= 2:
        A2 = A @ A
    if deg > 8:
        A3 = A @ A2
    if deg == 18:
        A6 = A3 @ A3

    if deg == 1:
        return I + A
    elif deg == 2:
        return I + A + .5 * A2
    elif deg == 4:
        return I + A + A2 @ (.5 * I + A / 6. + A2 / 24.)
    elif deg == 8:
        SQRT = np.sqrt(177.)
        x3 = 2. / 3.
        a1 = (1. + SQRT) * x3
        x1 = a1 / 88.
        x2 = a1 / 352.
        c0 = (-271. + 29. * SQRT) / (315. * x3)
        c1 = (11. * (-1. + SQRT)) / (1260. * x3)
        c2 = (11. * (-9. + SQRT)) / (5040. * x3)
        c4 = (89. - SQRT) / (5040. * x3 * x3)
        y2 = ((857. - 58. * SQRT)) / 630.
        A4 = A2 @ (x1 * A + x2 * A2)
        A8 = (x3 * A2 + A4) @ (c0 * I + c1 * A + c2 * A2 + c4 * A4)
        return I + A + y2 * A2 + A8
    elif deg == 12:
        b = ms.Tensor(
            np.array(
                [[-1.86023205146205530824e-02,
                  -5.00702322573317714499e-03,
                  -5.73420122960522249400e-01,
                  -1.33399693943892061476e-01],
                 [4.6,
                  9.92875103538486847299e-01,
                  -1.32445561052799642976e-01,
                  1.72990000000000000000e-03],
                 [2.11693118299809440730e-01,
                  1.58224384715726723583e-01,
                  1.65635169436727403003e-01,
                  1.07862779315792429308e-02],
                 [0.,
                  -1.31810610138301836924e-01,
                  -2.02785554058925905629e-02,
                  -6.75951846863086323186e-03]]),
            A.dtype)
        q = msnp.stack([I, A, A2, A3], axis=-3).unsqueeze(-4)
        len_batch = A.ndim - 2
        q_size = [-1 for _ in range(len_batch)] + [4, -1, -1, -1]
        q = ops.expand(q, *q_size)
        b = b.unsqueeze(-1).unsqueeze(-1).expand_as(q)
        msnp.sum(b * q, axis=-3)
        if batched:
            qaux = q[..., 2, :, :] + q[..., 3, :, :] @ q[..., 3, :, :]
            return q[..., 0, :, :] + (q[..., 1, :, :] + qaux) @ qaux
        else:
            qaux = q[2] + q[3] @ q[3]
            return q[0] + (q[1] + qaux) @ qaux
    elif deg == 18:
        b = ms.Tensor(
            np.array(
                [[0.,
                  -1.00365581030144618291e-01,
                  -8.02924648241156932449e-03,
                  -8.92138498045729985177e-04,
                  0.],
                 [0.,
                  3.97849749499645077844e-01,
                  1.36783778460411720168e+00,
                  4.98289622525382669416e-01,
                  -6.37898194594723280150e-04],
                 [-1.09676396052962061844e+01,
                  1.68015813878906206114e+00,
                  5.71779846478865511061e-02,
                  -6.98210122488052056106e-03,
                  3.34975017086070470649e-05],
                 [-9.04316832390810593223e-02,
                  -6.76404519071381882256e-02,
                  6.75961301770459654925e-02,
                  2.95552570429315521194e-02,
                  -1.39180257516060693404e-05],
                 [0.,
                  0.,
                  -9.23364619367118555360e-02,
                  -1.69364939002081722752e-02,
                  -1.40086798182036094347e-05]]),
            A.dtype)
        q = msnp.stack([I, A, A2, A3, A6], axis=-3).unsqueeze(-4)
        len_batch = A.ndimension() - 2
        q_size = [-1 for _ in range(len_batch)] + [5, -1, -1, -1]
        q = q.expand(ms.Tensor(q_size))
        b = b.unsqueeze(-1).unsqueeze(-1).expand_as(q)
        q = msnp.sum(b * q, axis=-3)
        if batched:
            qaux = q[..., 0, :, :] @ q[..., 4, :, :] + q[..., 3, :, :]
            return q[..., 1, :, :] + (q[..., 2, :, :] + qaux) @ qaux
        else:
            qaux = q[0] @ q[4] + q[3]
            return q[1] + (q[2] + qaux) @ qaux


def _expm_taylor(A: ms.Tensor):
    if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
        raise ValueError('Expected a square matrix or a batch of square matrices')
    if A.ndim == 2:
        if A.shape == (1, 1):
            return ops.exp(A)
        if A.size > 4:
            thetas = thetas_dict["double"]
        else:
            thetas = thetas_dict["single"]
        normA = ops.max(msnp.sum(ops.abs(A), axis=0))[1]
        for deg, theta in zip(degs, thetas):
            if normA <= theta:
                return _taylor_approx(A, deg)
        s = int(np.ceil(np.log2(normA) - np.log2(thetas[-1])))
        A = A * (2 ** -s)
        X = _taylor_approx(A, degs[-1])
        return msnp.matrix_power(X, 2 ** s)
    else:
        if A.shape[-2:] == (1, 1):
            return ops.exp(A)
        if A.size > 4:
            thetas = thetas_dict["double"]
        else:
            thetas = thetas_dict["single"]
        normA = ops.max(msnp.sum(msnp.abs(A), axis=-2), axis=-1)[1]
        if ops.all(normA == 0.):
            I = msnp.eye(A.shape[-2], A.shape[-1], dtype=A.dtype)
            I = I.expand_as(A)
            return I
        more = np.where(normA > thetas[-1])[0].tolist()
        k = ops.zeros_like(normA)
        k[more] = ops.ceil(msnp.log2(normA[more]) - np.log2(thetas[-1]))
        A = ops.pow(.5, k.float()).unsqueeze(-1).unsqueeze(-1).expand_as(A) * A
        X = _taylor_approx(A, degs[-1])
        return _matrix_power_two_batch(X, k)


def diagflat(
        x: Union[ms.Tensor, ms.Parameter],
        /,
        *,
        offset: Optional[int] = 0,
        padding_value: Optional[float] = 0.,
        align: Optional[str] = "RIGHT_LEFT",
        num_rows: Optional[int] = None,
        num_cols: Optional[int] = None,
        out: Optional[Union[ms.Tensor, ms.Parameter]] = None,
):
    if len(x.shape) > 1:
        a = msnp.ravel(x)
    else:
        a = x

    if num_rows is None:
        num_rows = -1
    if num_cols is None:
        num_cols = -1
    ret = ops.matrix_diag(
        a,
        k=offset,
        num_rows=ms.Tensor(num_rows, dtype=ms.int32),
        num_cols=ms.Tensor(num_cols, dtype=ms.int32),
        padding_value=ms.Tensor(padding_value),
        align=align,
    )

    return ret

def kron(
        a: ms.Tensor,
        b: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return ops.kron(a, b)


def matrix_exp(
        x: Union[ms.Tensor, ms.Parameter],
        /,
        *,
        out: Optional[Union[ms.Tensor, ms.Parameter]] = None,
) -> Union[ms.Tensor, ms.Parameter]:
    return _expm_taylor(x)


@with_unsupported_dtypes({"2.0.0 and below": ("Complex", )}, backend_version)
def eigvals(x: ms.Tensor, /) -> ms.Tensor:
    x = _cast_int_to_float(x)
    return ops.svd(x)[0]


@with_unsupported_dtypes({"2.0.0 and below": ("complex", )}, backend_version)
def adjoint(
        x: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    _check_valid_dimension_size(x)
    axes = list(range(len(x.shape)))
    axes[-1], axes[-2] = axes[-2], axes[-1]
    return ops.Conj()(msnp.transpose(x, axes=axes))


@with_unsupported_dtypes({"2.0.0 and below": ("int32", "int64")}, backend_version)
def multi_dot(
        x: Sequence[Union[ms.Tensor, ms.Parameter]],
        /,
        *,
        out: Optional[Union[ms.Tensor, ms.Parameter]] = None,
) -> ms.Tensor:
    if len(x) < 2:
        raise ValueError("Expecting at least two tensors.")
    dot_out = reduce(ops.matmul, x)
    return dot_out
