# global

from numbers import Number
from typing import Union, List, Optional, Sequence
import mindspore as ms
import mindspore.ops as ops
from mindspore import Type
from mindspore.ops import functional as F
import mindspore.numpy as msnp

# local
import ivy
from ivy.functional.backends.mindspore.device import _to_device
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
)
from .data_type import as_native_dtype
# Array API Standard #
# -------------------#



@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
def asarray(
    obj: Union[
        ms.Tensor, bool, int, float, NestedSequence, SupportsBufferProtocol
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Type = None,
    device: str,
    out: ms.Tensor = None,
) -> ms.Tensor:
    if copy:
        if dtype is None and isinstance(obj, ms.Tensor):
            return ops.identity(obj)
        if dtype is None and not isinstance(obj, ms.Tensor):
            try:
                dtype = ivy.default_dtype(item=obj, as_native=True)
                tensor = ms.Tensor(obj, dtype=dtype)
            except (TypeError, ValueError):
                dtype = ivy.default_dtype(dtype=dtype, item=obj, as_native=True)
                tensor = ms.Tensor(
                    ivy.nested_map(obj, lambda x: F.cast(x, dtype)),
                    dtype=dtype,
                )
            return ops.identity(F.cast(tensor, dtype))
        else:
            dtype = ivy.as_ivy_dtype(ivy.default_dtype(dtype=dtype, item=obj))
            try:
                tensor = ms.Tensor(obj, dtype=dtype)
            except (TypeError, ValueError):
                tensor = ms.Tensor(
                    ivy.nested_map(obj, lambda x: F.cast(x, dtype)),
                    dtype=dtype,
                )
            return ops.identity(F.cast(tensor, dtype))
    else:
        if dtype is None and isinstance(obj, ms.Tensor):
            return obj
        if dtype is None and not isinstance(obj, ms.Tensor):
            try:
                dtype = ivy.default_dtype(item=obj, as_native=True)
                return ms.Tensor(obj, dtype=dtype)
            except (TypeError, ValueError):
                dtype = ivy.as_ivy_dtype(ivy.default_dtype(dtype=dtype, item=obj))
                return ms.Tensor(
                    ivy.nested_map(obj, lambda x: F.cast(x, dtype)),
                    dtype=dtype,
                )
        else:
            dtype = ivy.default_dtype(dtype=dtype, item=obj, as_native=True)
            try:
                tensor = ms.Tensor(obj, dtype=dtype)
            except (TypeError, ValueError):
                print('no this ValueError', )
                tensor = ms.Tensor(
                    ivy.nested_map(obj, lambda x: F.cast(x, dtype)),
                    dtype=dtype,
                )
            return F.cast(tensor, dtype)

def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[Type] = None,
    device: str = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if dtype:
        dtype = as_native_dtype(dtype)
    res = _to_device(msnp.arange(start, stop, step, dtype=dtype), device=device)
    if not dtype:
        if res.dtype == ms.float64:
            return res.astype(ms.float32)
        elif res.dtype == ms.int64:
            return res.astype(ms.int32)
    return res


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: Type,
    device: str = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return _to_device(msnp.empty(shape, dtype), device=device)


def empty_like(
    x: ms.Tensor, /, *, dtype: Type, device: str = None, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    return _to_device(msnp.empty_like(x, dtype=dtype), device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: Type,
    device: str = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if n_cols is None:
        n_cols = n_rows
    i = msnp.eye(n_rows, n_cols, k, dtype)
    if batch_shape is None:
        return _to_device(i, device=device)
    else:
        reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
        tile_dims = list(batch_shape) + [1, 1]
        return_mat = msnp.tile(msnp.reshape(i, reshape_dims), tile_dims)
        return _to_device(return_mat, device=device)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, Type]] = None,
    device: str = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    dtype = ivy.default_dtype(dtype=dtype, item=fill_value, as_native=True)
    ivy.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    return _to_device(
        msnp.full(shape, fill_value, dtype),
        device=device,
    )


def full_like(
    x: ms.Tensor,
    /,
    fill_value: Number,
    *,
    dtype: Type,
    device: str = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    ivy.utils.assertions.check_fill_value_and_dtype_are_compatible(fill_value, dtype)
    return _to_device(msnp.full_like(x, fill_value, dtype=dtype), device=device)


def linspace(
    start: Union[ms.Tensor, float],
    stop: Union[ms.Tensor, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: Type,
    device: str = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if axis is None:
        axis = -1
    ans = msnp.linspace(start, stop, num, endpoint, dtype=dtype, axis=axis)
    if (
        ans.shape[0] >= 1
        and (not isinstance(start, ms.Tensor))
        and (not isinstance(stop, ms.Tensor))
    ):
        ans[0] = start
    return _to_device(ans, device=device)


def meshgrid(
    *arrays: ms.Tensor,
    sparse: bool = False,
    indexing: str = "xy",
) -> List[ms.Tensor]:
    return msnp.meshgrid(*arrays, sparse=sparse, indexing=indexing)


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: Type,
    device: str,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return _to_device(msnp.ones(shape, dtype), device=device)


def ones_like(
    x: ms.Tensor,
    /,
    *,
    dtype: Type,
    device: str = None,
    out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    return _to_device(msnp.ones_like(x, dtype=dtype), device=device)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: Type,
    device: str = None,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return _to_device(msnp.zeros(shape, dtype), device=device)


zeros.support_native_out = True


def zeros_like(
    x: ms.Tensor,
    /,
    *,
    dtype: Type,
    device: str = None,
    out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    return _to_device(msnp.zeros_like(x, dtype=dtype), device=device)


def tril(
    x: ms.Tensor, /, *, k: int = 0, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    return msnp.tril(x, k)


def triu(
    x: ms.Tensor, /, *, k: int = 0, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    return msnp.triu(x, k)


# Extra #
# ------#


def copy_array(
    x: ms.Tensor,
    *,
    to_ivy_array: Optional[bool] = True,
    out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if to_ivy_array:
        return ivy.to_ivy(x.copy())
    return x.copy()
