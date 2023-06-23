from numbers import Number
from typing import Optional, Tuple, Union, Sequence
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as msnp
import mindspore.ops.functional as F
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version
from mindspore._c_expression.typing import Float, Int, Complex
from mindspore import Type
from typing import (
    Optional,
    Union,
    Sequence,
    Tuple,
    List,
)

# local
import ivy
from ivy import promote_types_of_inputs

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


def moveaxis(
        a: ms.Tensor,
        source: Union[int, Sequence[int]],
        destination: Union[int, Sequence[int]],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.moveaxis(a, source, destination)


moveaxis.support_native_out = False


def heaviside(
        x1: ms.Tensor,
        x2: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.heaviside(
        x1,
        x2,
    )


heaviside.support_native_out = True


def flipud(
        m: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.flipud(m)


flipud.support_native_out = False


def vstack(
        arrays: Sequence[ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.vstack(arrays)


def hstack(
        arrays: Sequence[ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.hstack(arrays)


def rot90(
        m: ms.Tensor,
        /,
        *,
        k: Optional[int] = 1,
        axes: Optional[Tuple[int, int]] = (0, 1),
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.rot90(m, k, axes)


def fliplr(
        m: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.fliplr(m)


fliplr.support_native_out = False


def _chbevl(x, vals):
    b0 = vals[0]
    b1 = 0.0

    for i in range(1, len(vals)):
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + vals[i]

    return 0.5 * (b0 - b2)


_i0A = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
]

_i0B = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
]


def _i0_1(x):
    return msnp.exp(x) * _chbevl(x / 2.0 - 2, _i0A)


def _i0_2(x):
    return msnp.exp(x) * _chbevl(32.0 / x - 2.0, _i0B) / msnp.sqrt(x)


@with_unsupported_dtypes({"2.0.0 and below": ("complex",)}, backend_version)
def i0(
        x: ms.Tensor,
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    x = _cast_int_to_float(x)
    x = ops.abs(x)
    return msnp.piecewise(x, [x <= 8.0], [_i0_1, _i0_2])


def _array_split(ary, indices_or_sections, axis=0):
    try:
        Ntotal = ary.shape[axis]
    except AttributeError:
        Ntotal = len(ary)
    try:
        # handle array case.
        Nsections = len(indices_or_sections) + 1
        div_points = [0] + list(indices_or_sections) + [Ntotal]
    except TypeError:
        # indices_or_sections is a scalar, not an array.
        Nsections = int(indices_or_sections)
        if Nsections <= 0:
            raise ValueError('number sections must be larger than 0.') from None
        Neach_section, extras = divmod(Ntotal, Nsections)
        section_sizes = ([0] +
                         extras * [Neach_section + 1] +
                         (Nsections - extras) * [Neach_section])
        div_points = ms.Tensor(section_sizes, dtype=ms.int32).cumsum()

    sub_arys = []
    sary = msnp.swapaxes(ary, axis, 0)
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(msnp.swapaxes(sary[st:end], axis, 0))

    return sub_arys


@with_unsupported_dtypes({"2.0.0 and below": ("complex64", "complex128")}, backend_version)
def vsplit(
        ary: ms.Tensor,
        indices_or_sections: Union[int, Tuple[int, ...]],
        /,
) -> List[ms.Tensor]:
    return _array_split(ary, indices_or_sections)

@with_unsupported_dtypes({"2.0.0 and below": ("complex64", "complex128")}, backend_version)
def dsplit(
        ary: ms.Tensor,
        indices_or_sections: Union[int, Tuple[int, ...]],
        /,
) -> List[ms.Tensor]:
    if ary.ndim < 3:
        raise ivy.utils.exceptions.IvyError(
            "dsplit only works on arrays of 3 or more dimensions"
        )
    return _array_split(ary, indices_or_sections, 2)


def dstack(
        arrays: Sequence[ms.Tensor],
        /,
        *,
        out: Optional[ms.Tensor] = None,
) -> ms.Tensor:
    return msnp.dstack(arrays)


def atleast_1d(*arys: Union[ms.Tensor, bool, Number]) -> List[ms.Tensor]:
    return msnp.atleast_1d(*arys)


def atleast_2d(*arys: ms.Tensor) -> List[ms.Tensor]:
    return msnp.atleast_2d(*arys)


def atleast_3d(*arys: Union[ms.Tensor, bool, Number]) -> List[ms.Tensor]:
    return msnp.atleast_3d(*arys)
