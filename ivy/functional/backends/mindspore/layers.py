"""Collection of Mindspore network layers, wrapped to fit Ivy syntax and signature."""

from typing import Optional, Tuple, Union, Sequence, List

# global
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.numpy as msnp

# local
import ivy
from ivy.functional.ivy.layers import _handle_padding, _deconv_length
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def _pad_before_conv(x, filters, strides, padding, dims, dilations):
    dilations = [dilations] * dims if isinstance(dilations, int) else dilations
    strides = [strides] * dims if isinstance(strides, int) else strides
    if isinstance(padding, str):
        filter_shape = list(filters.shape[:dims])
        filter_shape = [
            filter_shape[i] + (filter_shape[i] - 1) * (dilations[i] - 1)
            for i in range(dims)
        ]
        new_pad = [
            _handle_padding(x.shape[1 + i], strides[i], filter_shape[i], padding)
            for i in range(dims)
        ]
        pad_list = [
            (new_pad[i] // 2, new_pad[i] - new_pad[i] // 2) for i in range(dims)
        ]
    else:
        pad_list = padding
    pad_list = tuple(pad_list)
    pad_op = ms.nn.Pad(paddings=
    (
        (0, 0),
        *pad_list,
        (0, 0),
    )
    )
    return pad_op(x)


@with_unsupported_dtypes(
    {"2.0.0 and below": ("float16", "bfloat16", "complex")},
    backend_version,
)
def conv1d(
        x: ms.Tensor,
        filters: ms.Tensor,
        strides: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = 'valid',
        /,
        *,
        data_format: str = "NWC",
        dilations: Union[int, Tuple[int, int]] = 1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    xdtype = x.dtype
    if xdtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if filters.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        filters = filters.astype(ms.float32)
    filters = ops.transpose(filters, (2, 1, 0))
    if isinstance(padding, str):
        if data_format == "NWC":
            x = ops.transpose(x, (0, 2, 1))
        padding = padding.lower()
        conv1d_fn = ms.nn.Conv1d(x.shape[1],
                                 filters.shape[0],
                                 kernel_size=filters.shape[-1],
                                 weight_init=filters,
                                 stride=strides,
                                 pad_mode=padding,
                                 dilation=dilations)
    elif isinstance(padding, (int, Tuple, List)):
        if data_format == "NWC":
            x = _pad_before_conv(x, filters, strides, padding, 1, dilations)
            x = ops.transpose(x, (0, 2, 1))
        else:
            x = _pad_before_conv(ops.transpose(x, (0, 2, 1)), filters, strides, padding, 1, dilations)
            x = ops.transpose(x, (0, 2, 1))

        conv1d_fn = ms.nn.Conv1d(x.shape[1],
                                 filters.shape[0],
                                 kernel_size=filters.shape[-1],
                                 weight_init=filters,
                                 stride=strides,
                                 pad_mode='valid',
                                 dilation=dilations)
    else:
        raise TypeError(f'padding is either str, int, or Tuple[int, int]. Got {type(padding)}')
    res = conv1d_fn(x)
    if data_format == "NWC":
        res = ops.transpose(res, (0, 2, 1))
    return res.astype(xdtype)


def conv1d_transpose(
        x: ms.Tensor,
        filters: ms.Tensor,
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
        data_format: str = "NWC",
        dilations: Union[int, Tuple[int]] = 1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    xdtype = x.dtype
    if data_format == "NWC":
        x = ops.transpose(x, (0, 2, 1))
    if xdtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if filters.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        filters = filters.astype(ms.float32)
    filters = ops.transpose(filters, (2, 1, 0))
    if isinstance(padding, str):
        padding = padding.lower()
        conv1d_trans_fn = ms.nn.Conv1dTranspose(x.shape[1],
                                                filters.shape[0],
                                                kernel_size=filters.shape[-1],
                                                weight_init=filters,
                                                stride=strides,
                                                pad_mode=padding,
                                                dilation=dilations)
    elif isinstance(padding, (int, Tuple, List)):
        padding = _to_proper_padding_shape(padding, 1)
        if isinstance(padding, List):
            padding = tuple(padding)
        conv1d_trans_fn = ms.nn.Conv1dTranspose(x.shape[1],
                                                filters.shape[0],
                                                kernel_size=filters.shape[-1],
                                                weight_init=filters,
                                                stride=strides,
                                                padding=padding,
                                                pad_mode='pad',
                                                dilation=dilations)
    else:
        raise TypeError(f'padding is either str, int, or Tuple[int, int]. Got {type(padding)}')
    res = conv1d_trans_fn(x)
    if data_format == "NWC":
        res = ops.transpose(res, (0, 2, 1))
    return res.astype(xdtype)


def _check_iter(x):
    try:
        _ = iter(x)
        return True
    except TypeError:
        return False


def _to_proper_padding_shape(padding, dim_conv):
    target_shape = {1: 3, 2: 4, 3: 6}
    if _check_iter(padding):
        padding_size = len(padding)
        if padding_size == 1:
            padding = (padding[0]) * target_shape[dim_conv]
        elif (padding_size == 2 and dim_conv == 2) or (padding_size == 3 and dim_conv == 3):
            padding = padding * 2
    elif isinstance(padding, int):
        padding = (padding,) * target_shape[dim_conv]
    else:
        raise ValueError(f'padding is either iter. or int but got {type(padding)}.')
    if isinstance(padding, List):
        padding = tuple(padding)
    return padding


def conv2d(
        x: ms.Tensor,
        filters: ms.Tensor,
        strides: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = 'valid',
        /,
        *,
        data_format: str = "NHWC",
        output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
        dilations: Union[int, Tuple[int, int]] = 1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    xdtype = x.dtype
    x = x.astype(ms.float32)
    filters = filters.astype(ms.float32)
    filters = ops.transpose(filters, (2, 3, 0, 1))
    if isinstance(padding, str):
        padding = padding.lower()
        if data_format == "NHWC":
            x = ops.transpose(x, (0, 3, 1, 2))
        conv2d_fn = ms.nn.Conv2d(x.shape[1],
                                 filters.shape[0],
                                 kernel_size=filters.shape[2:],
                                 weight_init=filters,
                                 stride=strides,
                                 pad_mode=padding,
                                 dilation=dilations,
                                 data_format='NCHW')
    elif isinstance(padding, (int, Tuple, List)):
        if data_format == "NHWC":
            x = _pad_before_conv(x, filters, strides, padding, 2, dilations)
            x = ops.transpose(x, (0, 3, 1, 2))
        else:
            x = _pad_before_conv(ops.transpose(x, (0, 2, 3, 1)), filters, strides, padding, 2, dilations)
            x = ops.transpose(x, (0, 3, 1, 2))

        conv2d_fn = ms.nn.Conv2d(x.shape[1],
                                 filters.shape[0],
                                 weight_init=filters,
                                 kernel_size=filters.shape[2:],
                                 stride=strides,
                                 pad_mode='valid',
                                 dilation=dilations,
                                 data_format='NCHW')
    else:
        raise TypeError(f'padding is either str, int, or Tuple[int, int]. Got type {type(padding)}')
    res = conv2d_fn(x)
    if data_format == "NHWC":
        res = ops.transpose(res, (0, 2, 3, 1))
    return res.astype(xdtype)


def conv2d_transpose(
        x: ms.Tensor,
        filters: ms.Tensor,
        strides: Union[int, Tuple[int]],
        padding: str,
        /,
        *,
        output_shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
        data_format: str = "NHWC",
        dilations: Union[int, Tuple[int]] = 1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    xdtype = x.dtype
    if data_format == "NHWC":
        x = ops.transpose(x, (0, 3, 1, 2))
    if xdtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if filters.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        filters = filters.astype(ms.float32)
    filters = ops.transpose(filters, (2, 3, 0, 1))
    if isinstance(padding, str):
        padding = padding.lower()
        conv2d_trans_fn = ms.nn.Conv2dTranspose(x.shape[1],
                                                filters.shape[0],
                                                kernel_size=filters.shape[2:],
                                                weight_init=filters,
                                                stride=strides,
                                                pad_mode=padding,
                                                dilation=dilations)
    elif isinstance(padding, (int, Tuple, List)):
        padding = _to_proper_padding_shape(padding, 2)
        if isinstance(padding, List):
            padding = tuple(padding)
        conv2d_trans_fn = ms.nn.Conv2dTranspose(x.shape[1],
                                                filters.shape[0],
                                                kernel_size=filters.shape[2:],
                                                weight_init=filters,
                                                stride=strides,
                                                padding=padding,
                                                pad_mode='pad',
                                                dilation=dilations)
    else:
        raise TypeError(f'padding is either str, int, or Tuple[int, int]. Got {type(padding)}')
    res = conv2d_trans_fn(x)
    if data_format == "NHWC":
        res = ops.transpose(res, (0, 2, 3, 1))
    return res.astype(xdtype)


def conv3d(
        x: ms.Tensor,
        filters: ms.Tensor,
        strides: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = 'valid',
        /,
        *,
        data_format: str = "NDHWC",
        dilations: Union[int, Tuple[int, int]] = 1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    xdtype = x.dtype
    if data_format == "NDHWC":
        x = ops.transpose(x, (0, 4, 1, 2, 3))
    if xdtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if filters.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        filters = filters.astype(ms.float32)
    filters = ops.transpose(filters, (4, 3, 0, 1, 2))
    if isinstance(padding, str):
        padding = padding.lower()
        conv3d_fn = nn.Conv3d(x.shape[1],
                              filters.shape[0],
                              kernel_size=filters.shape[2:],
                              pad_mode=padding,
                              stride=strides,
                              dilation=dilations,
                              weight_init=filters,
                              data_format='NCDHW')
    elif isinstance(padding, (int, Tuple, List)):
        padding = _to_proper_padding_shape(padding, 3)
        if isinstance(padding, List):
            padding = tuple(padding)
        conv3d_fn = nn.Conv3d(x.shape[1],
                              filters.shape[0],
                              kernel_size=filters.shape[2:],
                              pad_mode='pad',
                              padding=padding,
                              stride=strides,
                              dilation=dilations,
                              weight_init=filters,
                              data_format='NCDHW')
    else:
        raise TypeError(f'padding is either str, int, or Tuple[int, int]. Got type {type(padding)}')
    res = conv3d_fn(x)
    if data_format == "NDHWC":
        res = ops.transpose(res, (0, 2, 3, 4, 1))
    return res.astype(xdtype)


def conv3d_transpose(
        x: ms.Tensor,
        filters: ms.Tensor,
        strides: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, Sequence[Tuple[int, int]]] = 'valid',
        /,
        *,
        data_format: str = "NDHWC",
        dilations: Union[int, Tuple[int, int]] = 1,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    xdtype = x.dtype
    if data_format == "NDHWC":
        x = ops.transpose(x, (0, 4, 1, 2, 3))
    if x.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        x = x.astype(ms.float32)
    if filters.dtype in [ms.float64, ms.int8, ms.int16, ms.int32, ms.int64]:
        filters = filters.astype(ms.float32)
    filters = ops.transpose(filters, (3, 4, 0, 1, 2))
    if isinstance(padding, str):
        padding = padding.lower()
        conv3d_transpose_fn = nn.Conv3dTranspose(x.shape[1],
                                                 filters.shape[1],
                                                 kernel_size=filters.shape[2:],
                                                 pad_mode=padding,
                                                 stride=strides,
                                                 dilation=dilations,
                                                 weight_init=filters,
                                                 data_format='NCDHW')
    elif isinstance(padding, int) or isinstance(padding, Tuple, List):
        conv3d_transpose_fn = nn.Conv3dTranspose(x.shape[1],
                                                 filters.shape[1],
                                                 kernel_size=filters.shape[2:],
                                                 pad_mode='pad',
                                                 padding=padding,
                                                 stride=strides,
                                                 dilation=dilations,
                                                 weight_init=filters,
                                                 data_format='NCDHW')
    else:
        raise TypeError(f'padding is either str, int, or Tuple[int, int]. Got type {type(padding)}')
    res = conv3d_transpose_fn(x)
    if data_format == "NDHWC":
        res = ops.transpose(res, (0, 2, 3, 4, 1))
    return res.astype(xdtype)

