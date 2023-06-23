# global
from typing import Union

from mindspore import Type

# local
from ivy.functional.backends.torch import ivy_dtype_dict


def is_native_dtype(dtype_in: Union[Type, str], /) -> bool:
    if dtype_in in ivy_dtype_dict:
        return True
    else:
        return False
