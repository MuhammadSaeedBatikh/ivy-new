import mindspore as ms
import mindspore.ops as ops

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
        input = ops._get_cache_prim(ops.Transpose)()(input, new_order)

    # Handle the default case.
    x_shape = input.shape
    x_rank = ops.rank(input)
    if start_dim == 1 and end_dim == -1:
        if x_rank in (0, 1):
            return ops.reshape(input, (-1,))
        return ops._get_cache_prim(ops.Flatten)()(input)

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
