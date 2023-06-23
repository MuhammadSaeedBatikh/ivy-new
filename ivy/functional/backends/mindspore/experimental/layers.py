"""Collection of Mindspore network layers, wrapped to fit Ivy syntax and signature."""

from typing import Optional, Tuple, Union, Sequence

# global
import mindspore as ms
from mindspore.ops import operations as P
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.numpy as msnp
import numpy as np
import operator

# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def _check_maxpool_padding(padding, nd):
    if isinstance(padding, int):
        return (0,) * (3 - nd) + (padding,) * nd
    if isinstance(padding, (tuple, list)):
        if len(padding) == 1:
            return (0,) * (3 - nd) + tuple(padding * nd)
        if len(padding) != nd:
            raise ValueError(f"The length of padding must equal to {nd}, but got {len(padding)}.")
        return (0,) * (3 - nd) + tuple(padding)
    return padding


def _check_tuple_length(arg_name, prim_name, length):
    """check the tuple length"""
    if len(arg_name) != length:
        raise ValueError(f"The length of {prim_name} must be equal to {length}, "
                         f"but got {len(arg_name)}.")
    return arg_name


def _cal_dilation(dilation, nd):
    """check the dilation"""
    if isinstance(dilation, int):
        return dilation
    if isinstance(dilation, tuple):
        if len(dilation) == 1:
            return dilation[0]
        if len(dilation) == nd:
            return (3 - nd) * (1,) + dilation
        if nd == 1:
            raise ValueError(f"The length of 'dilation' must be 1, but got {len(dilation)}.")
        raise ValueError(f"The length of 'dilation' must be 1 or {nd}, but got {len(dilation)}.")
    raise ValueError(f"The 'dilation' must be int or tuple, but got {type(dilation)}.")


def _ms_max_pool2d_fn(
        kernel_size=1,
        stride=1,
        padding=0,
        pad_mode='valid',
        format: str = "NHWC",
        dilation=1,
        ceil_mode: bool = False):
    if pad_mode.upper() == 'PAD':
        if format == "NHWC":
            raise ValueError(f"The 'NHWC' data format are not support when 'pad_mode' is 'pad'.")
        if isinstance(kernel_size, tuple):
            _check_tuple_length(kernel_size, 'kernel_size', 2)
            kernel_size = (1,) + kernel_size
        elif isinstance(kernel_size, int):
            kernel_size = (1, kernel_size, kernel_size)
        if isinstance(stride, tuple):
            _check_tuple_length(stride, 'stride', 2, )
            stride = (1,) + stride
        elif isinstance(stride, int):
            stride = (1, stride, stride)
        padding = _check_maxpool_padding(padding, 2)
        dilation = _cal_dilation(dilation, 2)
        return P.MaxPool3DWithArgmax(ksize=kernel_size, strides=stride, pads=padding,
                                     dilation=dilation, ceil_mode=ceil_mode)
    else:
        if padding != 0 or dilation != 1 or ceil_mode:
            raise ValueError(f"The parameter 'padding', 'dilation', 'return_indices', 'ceil_mode' "
                             f"can not be set to non-default value when pad_mode is not 'pad', "
                             f"but got pad_mode:{pad_mode}.")
        return P.MaxPool(kernel_size=kernel_size,
                         strides=stride,
                         pad_mode=pad_mode,
                         data_format=format)


def max_pool2d(
        x: ms.Tensor,
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, int, Tuple[int], Tuple[int, int]],
        /,
        *,
        data_format: str = "NHWC",
        dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if x.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if isinstance(padding, str):
        maxpool2d_fn = _ms_max_pool2d_fn(kernel_size=kernel,
                                         stride=strides,
                                         pad_mode=padding,
                                         ceil_mode=ceil_mode,
                                         dilation=dilation,
                                         format='NCHW')
    elif isinstance(padding, int) or isinstance(padding, Tuple):
        maxpool2d_fn = _ms_max_pool2d_fn(kernel_size=kernel,
                                         stride=strides,
                                         ceil_mode=ceil_mode,
                                         dilation=dilation,
                                         padding=padding,
                                         format='NCHW')
    else:
        raise TypeError(f'padding is either str, int, or Tuple[int, int]. Got type {type(padding)}')
    res = maxpool2d_fn(x)
    if data_format == "NHWC":
        res = res.permute(0, 2, 3, 1)
    return res


def max_pool3d(
        x: ms.Tensor,
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, int, Tuple[int], Tuple[int, int]],
        /,
        *,
        data_format: str = "NHWC",
        dilation: Union[int, Tuple[int], Tuple[int, int]] = 1,
        ceil_mode: bool = False,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if dilation != 1:
        raise ValueError(f"`Dilation != 1 is not supported.")

    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 2)
    if x.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if isinstance(padding, str):
        if padding.upper() in ['VALID', 'SAME']:
            if ceil_mode:
                raise ValueError(f"`ceil_mode' only supports 'None' or `False` when padding is VALID or SAME")
            else:
                ceil_mode = None
        maxpool3d_fn = ops.MaxPool3D(kernel_size=kernel,
                                     strides=strides,
                                     pad_mode=padding,
                                     ceil_mode=ceil_mode,
                                     data_format='NCDHW')
    elif isinstance(padding, int) or isinstance(padding, Tuple):
        maxpool3d_fn = ops.MaxPool3D(kernel_size=kernel,
                                     strides=strides,
                                     ceil_mode=ceil_mode,
                                     pad_list=padding,
                                     pad_mode="pad",
                                     data_format='NCDHW')
    res = maxpool3d_fn(x)
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


def avg_pool1d(
        x: ms.Tensor,
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[int, str] = 'valid',
        /,
        *,
        data_format: str = "NWC",
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if data_format == "NWC":
        x = x.permute(0, 2, 1)
    if x.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if isinstance(padding, str):
        avgpool1d_fn = nn.AvgPool1d(kernel_size=kernel,
                                    stride=strides,
                                    pad_mode=padding)
        res = avgpool1d_fn(x)
    elif isinstance(padding, int) or isinstance(padding, Tuple):
        res = ops.avg_pool1d(x,
                             kernel_size=kernel,
                             stride=strides,
                             padding=padding)
    if data_format == "NWC":
        res = res.permute(0, 2, 1)
    return res


def avg_pool2d(
        x: ms.Tensor,
        kernel: Union[int, Tuple[int], Tuple[int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[int, str] = 'valid',
        /,
        *,
        data_format: str = "NHWC",
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if data_format == "NHWC":
        x = x.permute(0, 3, 1, 2)
    if x.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if isinstance(padding, str):
        avgpool2d_fn = ops.AvgPool(kernel_size=kernel,
                                   strides=strides,
                                   pad_mode=padding,
                                   data_format='NCHW')
        res = avgpool2d_fn(x)
    elif isinstance(padding, int) or isinstance(padding, Tuple):
        res = ops.avg_pool2d(x,
                             kernel_size=kernel,
                             stride=strides,
                             padding=padding)
    if data_format == "NHWC":
        res = res.permute(0, 2, 3, 1)
    return res


def avg_pool3d(
        x: ms.Tensor,
        kernel: Union[int, Tuple[int], Tuple[int, int, int]],
        strides: Union[int, Tuple[int], Tuple[int, int, int]],
        padding: Union[int, str] = 'valid',
        /,
        *,
        data_format: str = "NDHWC",
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if data_format == "NDHWC":
        x = x.permute(0, 4, 1, 2, 2)
    if x.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if isinstance(padding, str):
        avgpool3d_fn = ops.AvgPool3D(kernel_size=kernel,
                                     strides=strides,
                                     pad_mode=padding,
                                     data_format='NCDHW')
    elif isinstance(padding, int) or isinstance(padding, Tuple):
        avgpool3d_fn = ops.AvgPool3D(kernel_size=kernel,
                                     strides=strides,
                                     pad=padding,
                                     pad_mode="pad",
                                     data_format='NCDHW')
    res = avgpool3d_fn(x)
    if data_format == "NDHWC":
        res = res.permute(0, 2, 3, 4, 1)
    return res


def _get_forward_norm(n, norm):
    if not isinstance(n, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(n)}"
        )
    if n <= 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid data points {n}, expecting more than 1"
        )
    if norm != "backward" and norm != "ortho" and norm != "forward":
        raise ivy.utils.exceptions.IvyError(f"Unrecognized normalization mode {norm}")

    if norm is None or norm == "backward":
        return 1
    elif norm == "ortho":
        return msnp.sqrt(n)
    elif norm == "forward":
        return n

    raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                     '"ortho" or "forward".')


def _fft_norm(s: ms.Tensor, func_name: str, norm: str) -> ms.Tensor:
    if norm == "backward":
        return ms.Tensor(1)
    elif norm == "ortho":
        return msnp.sqrt(ops.prod(s)) if func_name.startswith('i') else 1 / msnp.sqrt(ops.prod(s))
    elif norm == "forward":
        return ops.prod(s) if func_name.startswith('i') else 1 / ops.prod(s)
    raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                     '"ortho" or "forward".')


def _fft1d(x):
    n = len(x)
    if n == 1:
        return x
    X_even, X_odd = _fft1d(x[0::2]), _fft1d(x[1::2])
    factor = msnp.exp(-2j * msnp.pi * msnp.arange(n) / n)
    return msnp.concatenate(
        [X_even + factor[:int(n / 2)] * X_odd,
         X_even + factor[int(n / 2):] * X_odd])


def _ms_fft(a, s, axes, norm, fn_name):
    arr = ms.Tensor(a)
    if s is not None:
        s = ms.Tensor(np.array(tuple(map(operator.index, s))))
        if ops.any(msnp.less(s, ms.Tensor(0))):
            raise ValueError("Shape should be non-negative.")

    if s is not None and axes is not None and len(s) != len(axes):
        # Same error as numpy.
        raise ValueError("Shape and axes have different lengths.")

    orig_axes = axes
    if axes is None:
        if s is None:
            axes = range(arr.ndim)
        else:
            axes = range(arr.ndim - len(s), arr.ndim)

    if len(axes) != len(set(axes)):
        raise ValueError(
            f"{fn_name} does not support repeated axes. Got axes {axes}.")

    if len(axes) > 3:
        raise ValueError(
            "%s only supports 1D, 2D, and 3D FFTs. "
            "Got axes %s with input rank %s." % (fn_name, orig_axes, arr.ndim))

    if orig_axes is not None:
        axes = tuple(range(arr.ndim - len(axes), arr.ndim))
        arr = msnp.moveaxis(arr, orig_axes, axes)

    if s is not None:
        in_s = list(arr.shape)
        for axis, x in zip(axes, s):
            in_s[axis] = x
        # Cropping
        arr = arr[tuple(map(slice, in_s))]
        # Padding
        arr = msnp.pad(arr, [(0, x - y) for x, y in zip(in_s, arr.shape)])
    else:
        s = [arr.shape[axis] for axis in axes]
    if len(s) > 1:
        raise ValueError(f'Only one axis is currently supported. Got axis {s}')
    transformed = [None] * arr.shape[s]
    for i in range(arr.shape[s]):
        flat = msnp.take(arr, msnp.arange(arr.shape[s]), axis=s).flatten()
        transformed1d = _fft1d(flat)
    if norm is not None:
        transformed *= _fft_norm(s, fn_name, norm)

    if orig_axes is not None:
        transformed = msnp.moveaxis(transformed, axes, orig_axes)
    return transformed


def fft(
        x: ms.Tensor,
        dim: int,
        /,
        *,
        norm: Optional[str] = "backward",
        n: Union[int, Tuple[int]] = None,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    if not isinstance(dim, int):
        raise ivy.utils.exceptions.IvyError(
            f"Expecting <class 'int'> instead of {type(dim)}"
        )
    if -len(x.shape) < dim < len(x.shape) - 1:
        raise ivy.utils.exceptions.IvyError(
            f"Invalid dim {dim}, expecting ranging"
            " from {-len(x.shape)} to {len(x.shape)-1}  "
        )
    return _ms_fftn(x, n, dim, norm)


def adaptive_avg_pool1d(input, output_size):
    return ops.adaptive_avg_pool1d(input, output_size)


def adaptive_avg_pool2d(input, output_size):
    return ops.adaptive_avg_pool2d(input, output_size)


def adaptive_avg_pool3d(input, output_size):
    return ops.adaptive_avg_pool3d(input, output_size)


def adaptive_max_pool1d(input, output_size):
    return ops.adaptive_max_pool1d(input, output_size)


def adaptive_max_pool2d(input, output_size):
    return ops.adaptive_max_pool2d(input, output_size)


def adaptive_max_pool3d(input, output_size):
    return ops.adaptive_max_pool3d(input, output_size)
