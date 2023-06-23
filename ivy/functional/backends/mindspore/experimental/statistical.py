from numbers import Number
from typing import Optional, Tuple, Union
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
from mindspore._c_expression.typing import Float, Int, Complex
from mindspore import Type
from utility import _flatten_along_axes

# local
import ivy
from ivy import promote_types_of_inputs

def _get_gamma_mask(shape, default_value, conditioned_value, where):
    out = msnp.full(shape, default_value)
    ops.copyto(out, conditioned_value, where=where, casting="unsafe")
    return out
def _discret_interpolation_to_boundaries(index, gamma_condition_fun):
    previous = msnp.floor(index)
    next = previous + 1
    gamma = index - previous
    res = _get_gamma_mask(shape=index.shape,
                          default_value=next,
                          conditioned_value=previous,
                          where=gamma_condition_fun(gamma, index)
                          ).astype(ms.int64)
    # Some methods can lead to out-of-bound integers, clip them:
    res[res < 0] = 0
    return res

def _inverted_cdf(n, quantiles):
    gamma_fun = lambda gamma, _: (gamma == 0)
    return _discret_interpolation_to_boundaries((n * quantiles) - 1,
                                                gamma_fun)
def _compute_virtual_index(n, quantiles, alpha: float, beta: float):
    return n * quantiles + (
            alpha + quantiles * (1 - alpha - beta)
    ) - 1


_QuantileMethods = dict(
    # --- HYNDMAN and FAN METHODS
    # Discrete methods
    inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: _inverted_cdf(n, quantiles),
        fix_gamma=lambda gamma, _: gamma,  # should never be called
    ),
    averaged_inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles: (n * quantiles) - 1,
        fix_gamma=lambda gamma, _: _get_gamma_mask(
            shape=gamma.shape,
            default_value=1.,
            conditioned_value=0.5,
            where=gamma == 0),
    ),
    closest_observation=dict(
        get_virtual_index=lambda n, quantiles: _closest_observation(n,
                                                                    quantiles),
        fix_gamma=lambda gamma, _: gamma,  # should never be called
    ),
    # Continuous methods
    interpolated_inverted_cdf=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0, 1),
        fix_gamma=lambda gamma, _: gamma,
    ),
    hazen=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0.5, 0.5),
        fix_gamma=lambda gamma, _: gamma,
    ),
    weibull=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 0, 0),
        fix_gamma=lambda gamma, _: gamma,
    ),
    # Default method.
    # To avoid some rounding issues, `(n-1) * quantiles` is preferred to
    # `_compute_virtual_index(n, quantiles, 1, 1)`.
    # They are mathematically equivalent.
    linear=dict(
        get_virtual_index=lambda n, quantiles: (n - 1) * quantiles,
        fix_gamma=lambda gamma, _: gamma,
    ),
    median_unbiased=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 1 / 3.0, 1 / 3.0),
        fix_gamma=lambda gamma, _: gamma,
    ),
    normal_unbiased=dict(
        get_virtual_index=lambda n, quantiles:
        _compute_virtual_index(n, quantiles, 3 / 8.0, 3 / 8.0),
        fix_gamma=lambda gamma, _: gamma,
    ),
    # --- OTHER METHODS
    lower=dict(
        get_virtual_index=lambda n, quantiles: np.floor(
            (n - 1) * quantiles).astype(np.intp),
        fix_gamma=lambda gamma, _: gamma,
        # should never be called, index dtype is int
    ),
    higher=dict(
        get_virtual_index=lambda n, quantiles: np.ceil(
            (n - 1) * quantiles).astype(np.intp),
        fix_gamma=lambda gamma, _: gamma,
        # should never be called, index dtype is int
    ),
    midpoint=dict(
        get_virtual_index=lambda n, quantiles: 0.5 * (
                np.floor((n - 1) * quantiles)
                + np.ceil((n - 1) * quantiles)),
        fix_gamma=lambda gamma, index: _get_gamma_mask(
            shape=gamma.shape,
            default_value=0.5,
            conditioned_value=0.,
            where=index % 1 == 0),
    ),
    nearest=dict(
        get_virtual_index=lambda n, quantiles: msnp.around(
            (n - 1) * quantiles).astype(ms.int64),
        fix_gamma=lambda gamma, _: gamma,
        # should never be called, index dtype is int
    ))


def median(
        input: ms.Tensor,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    # TODO: compute mean of two median values if even tensor.
    if axis is None:
        return input.flatten().median(keepdims=keepdims)[0]
    else:
        return input.median(axis=axis, keep_dims=keepdims)[0]


def corrcoef(
        x: ms.Tensor,
        /,
        *,
        y: Optional[ms.Tensor] = None,
        rowvar: Optional[bool] = True,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.corrcoef(x, y=y, rowvar=rowvar, dtype=x.dtype)


def nanmean(
        a: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Tuple[int]]] = None,
        keepdims: Optional[bool] = False,
        dtype: Optional[ms.Tensor] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if isinstance(axis, list):
        axis = tuple(axis)
    return msnp.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype)


def bincount(
        x: ms.Tensor,
        /,
        *,
        weights: Optional[ms.Tensor] = None,
        minlength: Optional[int] = 0,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if weights is not None:
        ret = msnp.bincount(x, weights=weights, minlength=minlength)
        ret = ret.astype(weights.dtype)
    else:
        ret = msnp.bincount(x, minlength=minlength)
        ret = ret.astype(x.dtype)
    return ret


def unravel_index(
        indices: ms.Tensor,
        shape: Tuple[int],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> Tuple:
    temp = indices.astype(ms.int32)
    ret = msnp.unravel_index(temp, shape)
    return tuple(ret)


def _quantile(
        arr: ms.Tensor,
        quantiles: ms.Tensor,
        axis: int = -1,
        method="linear",
        out=None,
):
    arr = ms.Tensor(arr)
    values_count = arr.shape[axis]
    DATA_AXIS = 0
    if axis != DATA_AXIS:  # But moveaxis is slow, so only call it if axis!=0.
        arr = msnp.moveaxis(arr, axis, destination=DATA_AXIS)
    try:
        method = _QuantileMethods[method]
    except KeyError:
        raise ValueError(
            f"{method!r} is not a valid method. Use one of: "
            f"{_QuantileMethods.keys()}") from None
    virtual_indexes = method["get_virtual_index"](values_count, quantiles)
    virtual_indexes = ms.Tensor(virtual_indexes)
    if msnp.issubdtype(virtual_indexes.dtype, np.integer):
        # No interpolation needed, take the points along axis
        if np.issubdtype(arr.dtype, np.inexact):
            # may contain nan, which would sort to the end
            arr.partition(concatenate((virtual_indexes.ravel(), [-1])), axis=0)
            slices_having_nans = np.isnan(arr[-1])
        else:
            # cannot contain nan
            arr.partition(virtual_indexes.ravel(), axis=0)
            slices_having_nans = np.array(False, dtype=bool)
        result = take(arr, virtual_indexes, axis=0, out=out)
    else:
        previous_indexes, next_indexes = _get_indexes(arr,
                                                      virtual_indexes,
                                                      values_count)
        # --- Sorting
        arr.partition(
            np.unique(np.concatenate(([0, -1],
                                      previous_indexes.ravel(),
                                      next_indexes.ravel(),
                                      ))),
            axis=DATA_AXIS)
        if np.issubdtype(arr.dtype, np.inexact):
            slices_having_nans = np.isnan(
                take(arr, indices=-1, axis=DATA_AXIS)
            )
        else:
            slices_having_nans = None
        # --- Get values from indexes
        previous = np.take(arr, previous_indexes, axis=DATA_AXIS)
        next = np.take(arr, next_indexes, axis=DATA_AXIS)
        # --- Linear interpolation
        gamma = _get_gamma(virtual_indexes, previous_indexes, method)
        result_shape = virtual_indexes.shape + (1,) * (arr.ndim - 1)
        gamma = gamma.reshape(result_shape)
        result = _lerp(previous,
                       next,
                       gamma,
                       out=out)
    if np.any(slices_having_nans):
        if result.ndim == 0 and out is None:
            # can't write to a scalar
            result = arr.dtype.type(np.nan)
        else:
            result[..., slices_having_nans] = np.nan
    return result

def _quantile_ureduce_func(
        a: ms.Tensor,
        q: ms.Tensor,
        axis: int = None,
        out=None,
        overwrite_input: bool = False,
        method="linear",
) -> ms.Tensor:
    if q.ndim > 2:
        raise ValueError("q must be a scalar or 1d")
    if overwrite_input:
        if axis is None:
            axis = 0
            arr = a.ravel()
        else:
            arr = a
    else:
        if axis is None:
            axis = 0
            arr = a.flatten()
        else:
            arr = a.copy()
    result = _quantile(arr,
                       quantiles=q,
                       axis=axis,
                       method=method,
                       out=out)
    return result

def _quantile_unchecked(a,
                        q,
                        axis=None,
                        out=None,
                        overwrite_input=False,
                        method="linear",
                        keepdims=False):
    """Assumes that q is in [0, 1], and is an ndarray"""
    return _ureduce(a,
                    func=_quantile_ureduce_func,
                    q=q,
                    keepdims=keepdims,
                    axis=axis,
                    out=out,
                    overwrite_input=overwrite_input,
                    method=method)

def _quantile_is_valid(q):
    if q.ndim == 1 and q.size < 10:
        for i in range(q.size):
            if not (0.0 <= q[i] <= 1.0):
                return False
    else:
        if not (ops.all(0 <= q) and ops.all(q <= 1)):
            return False
    return True


def ms_quantile(a,
              q,
              axis=None,
              out=None,
              overwrite_input=False,
              method="linear",
              keepdims=False,
              *,
              interpolation=None):
    if interpolation is not None:
        if method != "linear":
            raise TypeError(
                "You shall not pass both `method` and `interpolation`!\n"
                "(`interpolation` is Deprecated in favor of `method`)")
        method = interpolation

    q = ms.Tensor(q)
    if not _quantile_is_valid(q):
        raise ValueError("Quantiles must be in the range [0, 1]")
    return _quantile_unchecked(
        a, q, axis, out, overwrite_input, method, keepdims)


def quantile(
        a: ms.Tensor,
        q: Union[ms.Tensor, float],
        /,
        *,
        axis: Optional[Union[Sequence[int], int]] = None,
        keepdims: Optional[bool] = False,
        interpolation: Optional[str] = "linear",
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    temp = a.astype(ms.float64)
    if isinstance(q, ms.Tensor):
        qt = q.astype(ms.float64)
    else:
        qt = q
    if isinstance(axis, list) or isinstance(axis, tuple):
        dimension = a.ndim
        for x in axis:
            axis1 = x
            for axis2 in range(x + 1, dimension):
                temp = ops.transpose(temp, (axis1, axis2))
                axis1 = axis2
        temp = _flatten_along_axes(temp, start_dim=dimension - len(axis))
        return msnp.nan(
            temp, qt, dim=-1, keepdim=keepdims, interpolation=interpolation, out=out
        )
    return torch.quantile(
        temp, qt, dim=axis, keepdim=keepdims, interpolation=interpolation, out=out
    )
