# global
import mindspore as ms
import mindspore.ops as ops
from mindspore import Type
import mindspore.numpy as msnp
from . import backend_version
import numpy as ornp
from ivy.func_wrapper import with_unsupported_dtypes
from functools import wraps
from mindspore._c_expression.typing import Float, Int, Complex
from typing import Tuple, Optional
from collections import namedtuple


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


def _unique_flattened(x, return_counts=False, *, equal_nan=True):
    ms_sort = ops.Sort()
    x = x.flatten()
    x = ms_sort(x)[0]
    mask = ornp.empty(x.shape, dtype=ornp.bool_)
    mask[:1] = True
    if (equal_nan
            and x.shape[0] > 0
            and isinstance(x.dtype, (Complex, Float))
            and msnp.isnan(x[-1])):
        if isinstance(x.dtype, Complex):
            aux_first_nan = msnp.searchsorted(msnp.isnan(x), True, side='left')
        else:
            aux_first_nan = msnp.searchsorted(x, x[-1], side='left')
        if aux_first_nan > 0:
            mask[1:aux_first_nan] = (
                    x[1:aux_first_nan] != x[:aux_first_nan - 1])
        mask[aux_first_nan] = True
        mask[aux_first_nan + 1:] = False
    else:
        mask[1:] = x[1:] != x[:-1]
    mask_ms = ms.Tensor(ornp.argwhere(mask)).flatten()
    if return_counts:
        comb = ornp.nonzero(mask) + ([mask.size],)
        idx = ornp.concatenate(comb)
        counts_lis = ornp.diff(idx)
        return x[mask_ms], ms.Tensor(counts_lis)
    else:
        return x[mask_ms],


@_back_to_original_dtype('int')
def _unique(x, return_counts=False):
    if return_counts:
        ret = _unique_flattened(x, return_counts, equal_nan=True)
        if len(ret) == 1:
            return ret[0]
        else:
            return ret
    else:
        return ops.sort(msnp.unique(x))[0]


@with_unsupported_dtypes({"1.11.0 and below": ("float64", "float128")}, backend_version)
def unique_counts(x: ms.Tensor, /) -> Tuple[ms.Tensor, ms.Tensor]:
    v, c = _unique(ops.cast(x, ms.float32), return_counts=True)
    v = ops.cast(v, x.dtype)
    nan_idx = ornp.where(msnp.isnan(v))[0].tolist()
    c[nan_idx] = 1
    Results = namedtuple("Results", ["values", "counts"])
    return Results(v, c)


@with_unsupported_dtypes({"1.11.0 and below": ("float64", "float128")}, backend_version)
@_back_to_original_dtype(('int', 'float16'))
def unique_values(x: ms.Tensor, /, *, out: Optional[ms.Tensor] = None) -> ms.Tensor:
    return _unique(ops.cast(x, ms.float32), return_counts=False)
