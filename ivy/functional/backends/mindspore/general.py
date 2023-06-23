from typing import Optional, Union, List, Tuple
from numbers import Number
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
import numpy as orig_np
from typing import Callable, Sequence, Iterable

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes, to_native_arrays_and_back, inputs_to_native_arrays
from . import backend_version


def _check_iter(x):
    try:
        _ = iter(x)
        return True
    except TypeError:
        return False


def is_variable(x, /, *, exclusive=False):
    return isinstance(x, ms.Parameter)


def _to_device(x: ms.Tensor, device=None) -> ms.Tensor:
    """Private version of `to_device` to be used in backend implementations"""
    if device is not None:
        if "gpu" in device:
            raise ivy.utils.exceptions.IvyException(
                "Mindspore does not support GPU placement at the moment, "
            )
        elif "cpu" in device:
            pass
        else:
            raise ivy.utils.exceptions.IvyException(
                "Invalid device specified, must be in the form "
                "[ 'cpu:idx' | 'gpu:idx' ], but found {}".format(device)
            )
    return x


def isin(
        elements: ms.Tensor,
        test_elements: ms.Tensor,
        /,
        *,
        assume_unique: Optional[bool] = False,
        invert: Optional[bool] = False,
) -> ms.Tensor:
    # Mindspore argument assume_unique is not supported since the implementation does not
    # rely on the uniqueness of the input arrays.
    return msnp.isin(
        elements,
        test_elements,
        invert=invert,
    )


def is_native_array(x, /, *, exclusive=False):
    if isinstance(x, ms.Tensor):
        return True
    return False


def container_types():
    return []


def to_scalar(x: ms.Tensor, /) -> Number:
    if isinstance(x, Number):
        return x
    return x.item()


def get_item(
        x: ms.Tensor,
        /,
        query: Union[ms.Tensor, orig_np.ndarray, List[int], Tuple[int]],
) -> ms.Tensor:
    q = ms.Tensor(query) if isinstance(query, orig_np.ndarray) else query
    qtype = type(q[0]) if _check_iter(q) else type(q)
    if 'bool' in str(qtype).lower():
        q = msnp.where(q)
    return x.__getitem__(q)


def current_backend_str() -> str:
    return "mindspore"


def to_numpy(
        x: Union[ms.Tensor, List[ms.Tensor]], /, *, copy: bool = True
) -> Union[orig_np.ndarray, List[orig_np.ndarray]]:
    if isinstance(x, (float, int, bool)):
        return x
    elif isinstance(x, orig_np.ndarray):
        if copy:
            return x.copy()
        else:
            return x
    elif isinstance(x, ms.Tensor):
        if copy:
            return x.asnumpy().copy()
        else:
            return x.asnumpy()
    elif isinstance(x, list):
        return [ivy.to_numpy(u) for u in x]
    raise ivy.utils.exceptions.IvyException("Expected a Mindspore Tensor.")


def array_equal(x0: ms.Tensor, x1: ms.Tensor, /) -> bool:
    x0, x1 = ivy.promote_types_of_inputs(x0, x1)
    return ms.equal(x0, x1)


def vecdot(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        axis: int = -1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return msnp.tensordot(x1, x2, (axis, axis))


def shape(
        x: ms.Tensor,
        /,
        *,
        as_array: bool = False,
) -> Union[ivy.Shape, ivy.Array]:
    if as_array:
        return ivy.array(x.shape, dtype=ivy.default_int_dtype())
    else:
        return ivy.Shape(x.shape)


def get_num_dims(x: ms.Tensor,
                 /,
                 *,
                 as_array: bool = False) -> Union[ms.Tensor, int]:
    return ms.Tensor(x.ndim) if as_array else x.ndim


def inplace_arrays_supported():
    return True


def inplace_decrement(
        x: Union[ivy.Array, ms.Tensor],
        val: Union[ivy.Array, ms.Tensor],
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native.data -= val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_increment(
        x: Union[ivy.Array, ms.Tensor],
        val: Union[ivy.Array, ms.Tensor],
) -> ivy.Array:
    (x_native, val_native), _ = ivy.args_to_native(x, val)
    x_native += val_native
    if ivy.is_ivy_array(x):
        x.data = x_native
    else:
        x = ivy.Array(x_native)
    return x


def inplace_update(
        x: Union[ivy.Array, ms.Tensor],
        val: Union[ivy.Array, ms.Tensor],
        /,
        *,
        ensure_in_backend: bool = False,
) -> ivy.Array:
    ivy.utils.assertions.check_inplace_sizes_valid(x, val)
    if ivy.is_array(x) and ivy.is_array(val):
        (x_native, val_native), _ = ivy.args_to_native(x, val)
        if is_variable(x_native):
            x_native.data = val_native
        else:
            x_native[()] = val_native
        if ivy.is_ivy_array(x):
            x.data = x_native

        else:
            x = ivy.to_ivy(x_native)
        if ensure_in_backend:
            x._data = val_native
        return x
    else:
        return val


def inplace_variables_supported():
    return True


def to_list(x: ms.Tensor, /) -> list:
    if isinstance(x, orig_np.ndarray):
        return x.tolist()
    elif isinstance(x, ms.Tensor):
        return x.asnumpy().tolist()
    raise ivy.utils.exceptions.IvyException(f"Expected a Mindspore Tensor. Received {type(x)}")


def scatter_flat(
        indices: ms.Tensor,
        updates: ms.Tensor,
        /,
        *,
        size: Optional[int] = None,
        reduction: str = "sum",
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    target = out
    target_given = ivy.exists(target)
    indices = indices.astype(ms.int32)
    if ivy.exists(size) and ivy.exists(target):
        ivy.utils.assertions.check_equal(len(target.shape), 1)
        ivy.utils.assertions.check_equal(target.shape[0], size)
    if reduction == "sum":
        if not target_given:
            target = msnp.zeros([size], dtype=updates.dtype)
        target = ops.scatter_add(target, indices, updates)
    elif reduction == "replace":
        if not target_given:
            target = msnp.zeros([size], dtype=updates.dtype)
        target = ops.scatter_update(target, indices, updates)
    elif reduction == "min":
        if not target_given:
            target = msnp.ones([size], dtype=updates.dtype) * 1e12
        target = ops.scatter_min(target, indices, updates)
        if not target_given:
            target = msnp.where(target == 1e12, ms.Tensor(0.0), target)
    elif reduction == "max":
        if not target_given:
            target = msnp.ones([size], dtype=updates.dtype) * -1e12
        target = ops.scatter_max(target, indices, updates)
        if not target_given:
            target = msnp.where(target == -1e12, ms.Tensor(0.0), target)
    else:
        raise ivy.utils.exceptions.IvyException(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    return _to_device(target.astype(updates.dtype))


def scatter_nd(
        indices: ms.Tensor,
        updates: ms.Tensor,
        /,
        shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
        *,
        reduction="sum",
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    # parse numeric inputs
    if (
            indices not in [Ellipsis, ()]
            and not (isinstance(indices, Iterable) and Ellipsis in indices)
            and not isinstance(indices, slice)
            and not (
            isinstance(indices, Iterable) and any(isinstance(k, slice) for k in indices)
    )
    ):
        indices = [[indices]] if isinstance(indices, Number) else indices
        indices = ms.Tensor(indices)
        if len(indices.shape) < 2:
            indices = msnp.expand_dims(indices, 0)
    # keep below commented out, array API tests are passing without this
    # updates = [updates] if isinstance(updates, Number) else updates
    indices = indices.astype(ms.int32)

    # handle Ellipsis
    if isinstance(indices, tuple) or indices is Ellipsis or isinstance(indices, slice):
        indices_tuple = indices
    else:
        expected_shape = (
            indices.shape[:-1] + out.shape[indices.shape[-1]:]
            if ivy.exists(out)
            else indices.shape[:-1] + tuple(shape[indices.shape[-1]:])
        )
        if sum(updates.shape) < sum(expected_shape):
            updates = ivy.broadcast_to(updates, expected_shape)._data
        elif sum(updates.shape) > sum(expected_shape):
            indices = ivy.broadcast_to(
                indices, updates.shape[:1] + (indices.shape[-1],)
            )._data
        indices_flat = indices.reshape(-1, indices.shape[-1]).T
        indices_tuple = tuple(indices_flat) + (Ellipsis,)

    # implementation
    target = ms.Tensor(out, updates.dtype) if out is not None else out
    target_given = ivy.exists(target)
    if ivy.exists(shape) and ivy.exists(target):
        ivy.utils.assertions.check_equal(ivy.Shape(target.shape), ivy.Shape(shape))
    shape = list(shape) if ivy.exists(shape) else list(out.shape)
    if reduction == "sum":
        if not target_given:
            target = msnp.zeros(shape, dtype=updates.dtype)
        target = ops.scatter_nd_add(target, indices, updates)
    elif reduction == "replace":
        if not target_given:
            target = msnp.zeros(shape, dtype=updates.dtype)
        target = ops.scatter_nd_update(target, indices, updates)
    elif reduction == "min":
        if not target_given:
            target = msnp.ones(shape, dtype=updates.dtype) * 1e12
        target = ops.scatter_nd_min(target, indices, updates)
        if not target_given:
            target = msnp.asarray(
                msnp.where(target == 1e12, ms.Tensor(0.0), target), dtype=updates.dtype
            )
    elif reduction == "max":
        if not target_given:
            target = msnp.ones(shape, dtype=updates.dtype) * -1e12
        target = ops.scatter_nd_max(target, indices, updates)
        if not target_given:
            target = msnp.asarray(
                msnp.where(target == -1e12, ms.Tensor(0.0), target), dtype=updates.dtype
            )
    else:
        raise ivy.utils.exceptions.IvyException(
            'reduction is {}, but it must be one of "sum", "min" or "max"'.format(
                reduction
            )
        )
    print('indices', indices)
    print('indices', updates)
    print('target', target)
    if ivy.exists(out):
        return ivy.inplace_update(out, _to_device(target.astype(updates.dtype)))
    return _to_device(target.astype(updates.dtype))


def vmap(
        func: Callable,
        in_axes: Union[int, Sequence[int], Sequence[None]] = 0,
        out_axes: Optional[int] = 0,
) -> Callable:
    @ivy.to_native_arrays_and_back
    def _vmap(*args, **kwargs):

        # convert args tuple to list to allow mutability using moveaxis ahead.
        args = list(args)
        # if in_axis is a non-integer, its length should be equal to pos args.
        if isinstance(in_axes, (list, tuple)):
            ivy.utils.assertions.check_equal(
                len(args),
                len(in_axes),
                message="""in_axes should have a length equivalent to the number
                of positional arguments to the function being vectorized or it
                should be an integer""",
            )

        # checking axis_size consistency
        axis_size = set()

        if isinstance(in_axes, int):
            for arg in args:
                axis_size.add(arg.shape[in_axes])
        elif isinstance(in_axes, (list, tuple)):
            for arg, axis in zip(args, in_axes):
                if axis is not None:
                    axis_size.add(arg.shape[axis])

        if len(axis_size) > 1:
            raise ivy.utils.exceptions.IvyException(
                """Inconsistent sizes. All mapped axes should have the same size"""
            )

        # Making sure not all in_axes are None
        if isinstance(in_axes, (list, tuple)):
            ivy.utils.assertions.check_any(
                [ivy.exists(ax) for ax in in_axes],
                message="At least one of the axes should be specified (not None)",
            )
        else:
            ivy.utils.assertions.check_exists(
                in_axes, message="single value in_axes should not be None"
            )

        # Handling None in in_axes by broadcasting the axis_size
        if isinstance(in_axes, (tuple, list)) and None in in_axes:
            none_axis_index = list()
            for index, axis in enumerate(in_axes):
                if axis is None:
                    none_axis_index.append(index)

            for none_mapped_axis in none_axis_index:
                args[none_mapped_axis] = ops.broadcast_to(
                    args[none_mapped_axis],
                    (tuple(axis_size) + args[none_mapped_axis].shape)
                )

        # set up the axis to be mapped
        if isinstance(in_axes, (tuple, list)):
            for i in range(len(in_axes)):
                args[i] = msnp.moveaxis(args[i], in_axes[i], 0)
        elif isinstance(in_axes, int):
            args[0] = msnp.moveaxis(args[0], in_axes, 0)

        arr_results = []
        for arrays in zip(*args):
            single_op = func(*arrays)
            arr_results.append(single_op)
        res = ivy.stack(arr_results)
        if out_axes:
            res = ivy.moveaxis(res, 0, out_axes)
        return res

    return _vmap
