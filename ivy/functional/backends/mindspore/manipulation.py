# global
import mindspore as ms
from mindspore import Type
import mindspore.numpy as msnp
import mindspore.ops as ops
import numpy as ornp
from mindspore._c_expression.typing import Float, Int, Complex
from typing import Union, Optional, Tuple, List, Sequence, Iterable
from typing import Tuple, Optional
from collections import namedtuple

# local
import ivy


def _check_iter(x):
    try:
        _ = iter(x)
        return True
    except TypeError:
        return False


def _reshape_fortran_ms(x, shape):
    if len(x.shape) > 0:
        x = ops.transpose(x)
    return ops.transpose(ops.reshape(x, shape[::-1]))


def expand_dims(
        x: ms.Tensor,
        /,
        *,
        axis: Union[int, Sequence[int]] = 0,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return ops.expand_dims(x, axis)


def flip(
        x: ms.Tensor,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    num_dims = len(x.shape)
    if not num_dims:
        return x
    if axis is None:
        axis = list(range(num_dims))
    if type(axis) is int:
        axis = [axis]
    axis = [item + num_dims if item < 0 else item for item in axis]
    return msnp.flip(x, axis)


def permute_dims(
        x: ms.Tensor,
        /,
        axes: Tuple[int, ...],
        *,
        out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    if isinstance(axes, int):
        axes = (axes,)
    elif isinstance(axes, list):
        axes = tuple(axes)
    return ops.transpose(x, axes)


def roll(
        x: ms.Tensor,
        /,
        shift: Union[int, Sequence[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.roll(x, shift, axis)


def reshape(
        x: Union[ms.Tensor, ms.Parameter],
        /,
        shape: Union[ivy.NativeShape, Sequence[int]],
        *,
        copy: Optional[bool] = None,
        order: Optional[str] = "C",
        allowzero: Optional[bool] = True,
        out: Optional[Union[ms.Tensor, ms.Parameter]] = None,
) -> Union[ms.Tensor, ms.Parameter]:
    ivy.utils.assertions.check_elem_in_list(order, ["C", "F"])
    if not allowzero:
        shape = tuple(
            new_s if con else old_s
            for new_s, con, old_s in zip(shape, ms.Tensor(shape) != 0, x.shape)
        )
    else:
        shape = tuple(shape)
    if copy:
        newarr = msnp.copy(x)
        if order == "F":
            return _reshape_fortran_ms(newarr, shape)
        return ops.reshape(newarr, shape)
    if order == "F":
        return _reshape_fortran_ms(x, shape)
    return ops.reshape(x, shape)


def squeeze(
        x: ms.Tensor,
        /,
        axis: Union[int, Sequence[int]],
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if isinstance(axis, list):
        axis = tuple(axis)
    if x.shape == ():
        if axis is None or axis == 0 or axis == -1:
            return x
        raise ivy.utils.exceptions.IvyException(
            "tried to squeeze a zero-dimensional input by axis {}".format(axis)
        )
    return msnp.squeeze(x, axis=axis)


def stack(
        arrays: Union[Tuple[ms.Tensor], List[ms.Tensor]],
        /,
        *,
        axis: int = 0,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.stack(arrays, axis)


def swapaxes(
        x: ms.Tensor, axis0: int, axis1: int, /, *, out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    return msnp.swapaxes(x, axis0, axis1)


def concat(
        xs: Union[Tuple[ms.Tensor, ...], List[ms.Tensor]],
        /,
        *,
        axis: Optional[int] = 0,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    is_tuple = isinstance(tuple)
    if axis is None:
        if is_tuple:
            xs = list(xs)
        for i in range(len(xs)):
            if xs[i].shape == ():
                xs[i] = msnp.ravel(xs[i])
        if is_tuple:
            xs = tuple(xs)
    ret = msnp.concatenate(xs, axis)
    highest_dtype = xs[0].dtype
    for i in xs:
        highest_dtype = ivy.as_native_dtype(ivy.promote_types(highest_dtype, i.dtype))
    return ivy.astype(ret, highest_dtype, copy=False)


def tile(
        x: ms.Tensor,
        /,
        repeats: Sequence[int],
        *,
        out: Optional[ms.Tensor] = None
) -> ms.Tensor:
    return msnp.tile(x, repeats)
