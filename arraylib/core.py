from __future__ import annotations
import functools as ft
import cffi
import os
from pathlib import Path
import typing as tp
import math
from typing_extensions import Self
from io import StringIO
import operator as op
import dataclasses as dc

## -------------------------------------------------------------------------------------------------
## LIB
## -------------------------------------------------------------------------------------------------


ffi = cffi.FFI()

with open(os.path.join(Path(__file__).parent, "src", "arraylib.h")) as f:
    lines = f.read().splitlines()

ffi.cdef("\n".join(lines[lines.index("// START") : lines.index("// END")]))

# Define the C source and build options
ffi.set_source(
    "arraylib",  # Name of the generated module
    """
    #include "arraylib.h"
    """,
    include_dirs=["."],
)

lib = ffi.dlopen(os.path.join(Path(__file__).parent, "src", "arraylib.so"))

## -------------------------------------------------------------------------------------------------
## ARRAY
## -------------------------------------------------------------------------------------------------


@dc.dataclass
class Layout:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    ndim: int


class NDArray:
    """Multi-dimensional array"""

    def __init__(self, buffer):
        self.buffer = buffer

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.buffer.lay.shape[i] for i in range(self.ndim))

    @property
    def stride(self) -> tuple[int, ...]:
        return tuple(self.buffer.lay.stride[i] for i in range(self.ndim))

    @property
    def size(self) -> int:
        return self.buffer.data.size

    @property
    def view(self) -> bool:
        return self.buffer.view

    @property
    def layout(self) -> Layout:
        return Layout(self.shape, self.stride, self.ndim)

    @property
    def ndim(self) -> int:
        return self.buffer.lay.ndim

    def reshape(self, shape: tp.Sequence[int]) -> Self:
        return reshape(self, shape)

    def sum(self, *, dims=None) -> Self:
        return reduce_sum(self, dims=dims)

    def max(self, *, dims=None) -> Self:
        return reduce_max(self, dims=dims)

    def min(self, *, dims=None) -> Self:
        return reduce_min(self, dims=dims)

    def where(self, lhs, rhs) -> Self:
        return where(self, lhs, rhs)

    def cat(self, arrays, *, dims: tp.Sequence[int]) -> Self:
        return cat([self] + arrays, dims)

    def apply(self, fn) -> Self:
        return apply(fn, self)

    def transpose(self, dst: tp.Sequence[int]) -> Self:
        return transpose(self, dst)

    def move_dim(self, src: tp.Sequence[int], dst: tp.Sequence[int]) -> Self:
        return move_dim(self, src, dst)

    def ravel(self) -> Self:
        return ravel(self)

    @property
    def T(self) -> Self:
        return self.transpose(list(range(self.ndim - 1, -1, -1)))

    # magic methods

    def __pow__(self, other) -> Self:
        return pow(self, other)

    def __del__(self) -> None:
        free(self)

    def __getitem__(self, index) -> float | Self:
        return getitem(self, index)

    def __setitem__(self, index, value) -> Self:
        return setitem(self, index, value)

    def __mul__(self, other) -> Self:
        return mul(self, other)

    def __rmul__(self, other) -> Self:
        return mul(other, self)

    def __neg__(self) -> Self:
        return neg(self)

    def __add__(self, other) -> Self:
        return add(self, other)

    def __radd__(self, other) -> Self:
        return add(other, self)

    def __sub__(self, other) -> Self:
        return sub(self, other)

    def __rsub__(self, other) -> Self:
        return sub(other, self)

    def __truediv__(self, other) -> Self:
        return div(self, other)

    def __rtruediv__(self, other) -> Self:
        return div(other, self)

    def __lt__(self, other) -> Self:
        return lt(self, other)

    def __le__(self, other) -> Self:
        return leq(self, other)

    def __gt__(self, other) -> Self:
        return gt(self, other)

    def __ge__(self, other) -> Self:
        return geq(self, other)

    def __eq__(self, other) -> Self:
        return eq(self, other)

    def __ne__(self, other) -> Self:
        return neq(self, other)

    def __matmul__(self, other) -> Self:
        return matmul(self, other)

    def __str__(self) -> Self:
        return ppstr(self)

    def __repr__(self) -> Self:
        return pprepr(self)

    def __copy__(self) -> Self:
        return copy(self)


## -------------------------------------------------------------------------------------------------
## UTILS
## -------------------------------------------------------------------------------------------------


def slice_to_range(
    shape: tuple[int],
    ndim: int,
    slices: tp.Sequence[slice],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    assert isinstance(slices, tp.Sequence)
    assert all(isinstance(si, slice) for si in slices)
    assert len(slices) == ndim
    start = tuple(si.start or 0 for si in slices)
    end = tuple(si.stop or shape[i] for i, si in enumerate(slices))
    step = tuple(si.step or 1 for si in slices)
    shape = shape
    assert all(0 <= si < ei <= shape[i] for i, (si, ei) in enumerate(zip(start, end)))
    return start, end, step


def normalize_index(index: tp.Sequence[tp.Any] | int) -> tp.TypeGuard[tuple[tp.Any]]:
    return tuple(index) if isinstance(index, tp.Sequence) else (index,)


def is_broadcastable(lhs: Layout, rhs: Layout) -> bool:
    li = lhs.ndim - 1
    ri = rhs.ndim - 1
    while li >= 0 and ri >= 0:
        if (
            (lhs.shape[li] != rhs.shape[ri])
            and lhs.shape[li] != 1
            and rhs.shape[ri] != 1
        ):
            return False
        li -= 1
        ri -= 1
    return True


## -------------------------------------------------------------------------------------------------
## INITIALIZATION
## -------------------------------------------------------------------------------------------------


def array(elems: list[tp.Any]) -> NDArray:
    assert isinstance(elems, tp.Sequence)
    shape = {}

    def flatten(elems, dim):
        if not isinstance(elems, tp.Sequence):
            assert isinstance(elems, (float, int)), f"{elems=}"
            return [elems]
        if dim in shape:
            msg = f"{dim=} mismatch: {shape[dim]=}!={len(elems)=}"
            assert shape[dim] == len(elems), msg
        else:
            shape[dim] = len(elems)
        flat = []
        for elem in elems:
            flat.extend(flatten(elem, dim + 1))
        return flat

    flat = tuple(flatten(elems, 0))
    shape = tuple(shape.values())
    ndim = len(shape)
    shape = ffi.new("size_t[]", shape)
    elems = ffi.new("float[]", flat)
    array = lib.array_fill(elems, shape, ndim)
    return NDArray(array)


def ones(shape: tp.Sequence[int]):
    assert isinstance(shape, tp.Sequence)
    assert all(isinstance(i, int) for i in shape)
    shape = tuple(shape)
    ndim = len(shape)
    shape = ffi.new("size_t[]", shape)
    return NDArray(lib.array_ones(shape, ndim))


def zeros(shape: tp.Sequence[int]):
    assert isinstance(shape, tp.Sequence)
    assert all(isinstance(i, int) for i in shape)
    shape = tuple(shape)
    ndim = len(shape)
    shape = ffi.new("size_t[]", shape)
    return NDArray(lib.array_zeros(shape, ndim))


def arange(start: int, end: int | None = None, step: int = 1):
    """Create a range array

    Args:
        start (int): start value
        end (int, optional): end value. Defaults to None.
        step (int, optional): step value. Defaults to 1.
    """
    if end is None:
        end = start
        start = 0
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)
    assert (0 <= start < end) and (step > 0)
    return NDArray(lib.array_arange(start, end, step))


def linspace(start: int, end: int, n: int):
    """Create a linearly spaced array

    Args:
        start (int): start value
        end (int): end value
        n (int): number of points
    """
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(n, int)
    assert (0 <= start < end) and (n > 0)
    return NDArray(lib.array_linspace(start, end, n))


def free(array):
    lib.array_free(array.buffer)
    del array


## -------------------------------------------------------------------------------------------------
## INTEROPERABILITY
## -------------------------------------------------------------------------------------------------


def from_numpy(array):
    """Convert a numpy array to an arraylib array"""
    import numpy as np

    assert isinstance(array, np.ndarray), f"expected numpy array, got {type(array)=}"
    assert array.dtype == np.float32, f"expected float32 dtype, got {array.dtype=}"
    buffer = ffi.new("float[]", array.size)
    ffi.memmove(buffer, np.frombuffer(array), array.size * 4)
    size = ffi.new("size_t[]", (array.size,))
    al_array = NDArray(lib.array_fill(buffer, size, 1))
    al_array = reshape(al_array, array.shape)
    return al_array


def to_numpy(array: NDArray):
    """Convert an arraylib array to a numpy array"""
    import numpy as np

    return np.ndarray(
        buffer=ffi.buffer(array.buffer.data.mem, array.size * 4),
        shape=array.shape,
        dtype=np.float32,
        strides=[s * 4 for s in array.stride],
    )


## -------------------------------------------------------------------------------------------------
## ARRAY-ARRAY/ARRAY-SCALAR OPERATIONS
## -------------------------------------------------------------------------------------------------


def generate_binary_op(op):
    op = ffi.callback("float(float, float)")(op)

    def fn(lhs, rhs):
        if isinstance(lhs, NDArray) and isinstance(rhs, NDArray):
            assert is_broadcastable(lhs.layout, rhs.layout), "cannot broadcast shapes"
            return NDArray(lib.array_array_scalar_op(op, lhs.buffer, rhs.buffer))
        if isinstance(lhs, NDArray) and isinstance(rhs, (int, float)):
            return NDArray(lib.array_scalar_op(op, lhs.buffer, rhs))
        raise NotImplementedError(f"Not supported for {type(lhs)=} and {type(rhs)=}")

    return ft.wraps(op)(fn)


add = generate_binary_op(op.add)
sub = generate_binary_op(op.sub)
mul = generate_binary_op(op.mul)
div = generate_binary_op(op.truediv)
pow = generate_binary_op(op.pow)


def matmul(lhs, rhs):
    """Matrix multiply two arrays"""
    assert lhs.shape[-1] == rhs.shape[0]
    assert lhs.ndim == 2 and rhs.ndim == 2
    return NDArray(lib.array_array_matmul(lhs.buffer, rhs.buffer))


def dot(lhs, rhs):
    """Dot product of two 1D arrays"""
    assert isinstance(lhs, NDArray) and isinstance(rhs, NDArray)
    assert lhs.ndim == 1 and rhs.ndim == 1, "dot product requires 1D arrays"
    return NDArray(lib.array_array_dot(lhs.buffer, rhs.buffer))


## -------------------------------------------------------------------------------------------------
## ELEMENTWISE OPERATIONS
## -------------------------------------------------------------------------------------------------


def generate_unary_op(op):
    op = ffi.callback("float(float)")(op)

    def wrapper(array):
        return NDArray(lib.array_op(op, array.buffer))

    return ft.wraps(op)(wrapper)


log = generate_unary_op(math.log)
neg = generate_unary_op(lambda x: -x)
exp = generate_unary_op(math.exp)


def apply(fn, array):
    """Apply an elementwise function to an array"""
    assert callable(fn)
    fn = ffi.callback("float(float)")(fn)
    assert isinstance(array, NDArray)
    return NDArray(lib.array_op(fn, array.buffer))


## -------------------------------------------------------------------------------------------------
## RESHAPING
## -------------------------------------------------------------------------------------------------


def reshape(array, shape: tp.Sequence[int]):
    assert isinstance(shape, tp.Sequence)
    assert all(isinstance(i, int) for i in shape)
    shape = tuple(shape)
    dst_total = ft.reduce(lambda x, y: x * y, shape, 1)
    src_total = ft.reduce(lambda x, y: x * y, array.shape, 1)
    assert src_total == dst_total, f"incorrect reshape {shape=}!={array.shape=}"
    ndim = len(shape)
    shape = ffi.new("size_t[]", shape)
    return NDArray(lib.array_reshape(array.buffer, shape, ndim))


def transpose(array, dst: tp.Sequence[int]):
    assert isinstance(dst, tp.Sequence)
    assert all(isinstance(i, int) for i in dst)
    dst = tuple(dst)
    assert len(dst) == array.ndim
    dst = dst = ffi.new("size_t[]", dst)
    return NDArray(lib.array_transpose(array.buffer, dst))


def move_dim(array, src: tp.Sequence[int], dst: tp.Sequence[int]):
    assert isinstance(array, NDArray)
    assert isinstance(src, tp.Sequence)
    assert all(isinstance(i, int) for i in src)
    assert isinstance(dst, tp.Sequence)
    assert all(isinstance(i, int) for i in dst)
    src = tuple(src)
    dst = tuple(dst)
    assert len(src) == len(dst)
    assert 0 < len(src) <= array.ndim
    src = ffi.new("size_t[]", src)
    dst = ffi.new("size_t[]", dst)
    return NDArray(lib.array_move_dim(array.buffer, src, dst, len(src)))


def ravel(array):
    assert isinstance(array, NDArray)
    return NDArray(lib.array_ravel(array.buffer))


## -------------------------------------------------------------------------------------------------
## COMPARISON OPERATIONS
## -------------------------------------------------------------------------------------------------


eq = generate_binary_op(op.eq)
neq = generate_binary_op(op.ne)
leq = generate_binary_op(op.le)
lt = generate_binary_op(op.lt)
geq = generate_binary_op(op.ge)
gt = generate_binary_op(op.gt)

## -------------------------------------------------------------------------------------------------
## GETTERS
## -------------------------------------------------------------------------------------------------


def get_view_from_range(
    array,
    start: tuple[int, ...],
    end: tuple[int, ...],
    step: tuple[int, ...],
):
    assert isinstance(array, NDArray)
    assert isinstance(start, tuple)
    assert isinstance(end, tuple)
    assert isinstance(step, tuple)
    assert all(isinstance(i, int) for i in start)
    assert all(isinstance(i, int) for i in end)
    assert all(isinstance(i, int) for i in step)
    assert len(start) == len(end) == len(step) == array.ndim
    start = ffi.new("size_t[]", start)
    end = ffi.new("size_t[]", end)
    step = ffi.new("size_t[]", step)
    return NDArray(lib.array_get_view_from_range(array.buffer, start, end, step))


def get_scalar_from_index(array, index: tp.Sequence[int]) -> int:
    assert isinstance(array, NDArray)
    assert isinstance(index, tp.Sequence)
    assert all(isinstance(i, int) for i in index)
    assert len(index) == array.ndim
    assert all(idx < array.shape[i] for i, idx in enumerate(index))
    index = tuple(index)
    index = ffi.new("size_t[]", index)
    return lib.array_get_scalar_from_index(array.buffer, index)


def getitem(array, index: int | slice | tp.Sequence[int | slice]):
    index = normalize_index(index)
    assert len(index) == array.ndim, f"{len(index)} != {array.ndim}"
    if all(isinstance(i, int) for i in index):
        return get_scalar_from_index(array, index)
    if all(isinstance(i, slice) for i in index):
        start, end, step = slice_to_range(array.shape, array.ndim, index)
        return get_view_from_range(array, start, end, step)
    raise NotImplementedError(f"Index type not supported: {index}")


## -------------------------------------------------------------------------------------------------
## SETTERS
## -------------------------------------------------------------------------------------------------


def set_scalar_from_index(array, value: float, index: tp.Sequence[int]):
    assert isinstance(index, tp.Sequence)
    assert all(isinstance(i, int) for i in index)
    assert len(index) == array.ndim
    assert all(idx < array.shape[i] for i, idx in enumerate(index))
    index = tuple(index)
    index = ffi.new("size_t[]", index)
    lib.array_set_scalar_from_index(array.buffer, value, index)
    return array


def set_scalar_from_range(
    array,
    start: tuple[int, ...],
    end: tuple[int, ...],
    step: tuple[int, ...],
    value: float,
):
    assert isinstance(start, tuple)
    assert isinstance(end, tuple)
    assert isinstance(step, tuple)
    assert all(isinstance(i, int) for i in start)
    assert all(isinstance(i, int) for i in end)
    assert all(isinstance(i, int) for i in step)
    assert isinstance(value, float), type(value)
    assert len(start) == len(end) == len(step) == array.ndim
    shape = array.shape
    assert all(0 <= si < ei <= shape[i] for i, (si, ei) in enumerate(zip(start, end)))
    start = ffi.new("size_t[]", start)
    end = ffi.new("size_t[]", end)
    step = ffi.new("size_t[]", step)
    lib.array_set_scalar_from_range(array.buffer, start, end, step, value)
    return array


def set_view_from_array(
    array,
    start: tuple[int, ...],
    end: tuple[int, ...],
    step: tuple[int, ...],
    value: NDArray,
):
    assert isinstance(start, tuple)
    assert isinstance(end, tuple)
    assert isinstance(step, tuple)
    assert all(isinstance(i, int) for i in start)
    assert all(isinstance(i, int) for i in end)
    assert all(isinstance(i, int) for i in step)
    assert len(start) == len(end) == len(step) == value.ndim
    start = ffi.new("size_t[]", start)
    end = ffi.new("size_t[]", end)
    step = ffi.new("size_t[]", step)
    lib.array_set_view_from_array(array.buffer, value.buffer, start, end, step)
    return array


def setitem(
    array,
    index: int | slice | tp.Sequence[int | slice],
    value: float | NDArray | tp.Any,
):
    assert isinstance(array, NDArray)
    index: tuple[tp.Any] = normalize_index(index)
    assert len(index) == array.ndim, "full index required to match ndim"

    if all(isinstance(i, int) for i in index):
        assert isinstance(value, (float, int)), f"{type(value)=}"
        return set_scalar_from_index(array, value, index)

    if all(isinstance(i, slice) for i in index):
        start, end, step = slice_to_range(array.shape, array.ndim, index)
        if isinstance(value, (float, int)):
            return set_scalar_from_range(array, start, end, step, value)
        if isinstance(value, NDArray):
            start, end, step = slice_to_range(array.shape, array.ndim, index)
            return set_view_from_array(array, start, end, step, value)
        raise NotImplementedError(f"Value type not supported: {value}")
    raise NotImplementedError(f"Index type not supported: {index}")


## -------------------------------------------------------------------------------------------------
## COPY
## -------------------------------------------------------------------------------------------------


def copy(array, deep=False):
    array = array.buffer
    if deep:
        return NDArray(lib.array_deep_copy(array))
    return lib.array_shallow_copy(array)


## -------------------------------------------------------------------------------------------------
## REDUCTIONS OPERATIONS
## -------------------------------------------------------------------------------------------------


def reduce(fn, array, dims=None, init=0.0):
    """Reduce an array along an dim/axes with a python function"""
    assert isinstance(array, NDArray)
    assert isinstance(dims, tp.Sequence) or dims is None
    dims = tuple(range(array.ndim)) if dims is None else tuple(dims)
    assert 0 <= len(dims) <= array.ndim, "dim out of bounds"

    @ffi.callback("float(float, float)")
    def wrapped(x, y):
        return fn(x, y)

    dims = ffi.new("size_t[]", dims)
    return NDArray(lib.array_reduce(wrapped, array.buffer, dims, len(dims), init))


reduce_sum = ft.partial(reduce, lambda x, y: x + y)
reduce_max = ft.partial(reduce, max, init=float("-inf"))
reduce_min = ft.partial(reduce, min, init=float("inf"))


## -------------------------------------------------------------------------------------------------
## CONDITIONAL OPERATIONS
## -------------------------------------------------------------------------------------------------


def where(cond, lhs, rhs):
    assert isinstance(cond, NDArray)

    if isinstance(lhs, NDArray) and isinstance(rhs, NDArray):
        assert cond.shape == lhs.shape == rhs.shape
        return NDArray(lib.array_array_array_where(cond.buffer, lhs.buffer, rhs.buffer))
    if isinstance(lhs, NDArray) and isinstance(rhs, (int, float)):
        assert cond.shape == lhs.shape
        return NDArray(lib.array_array_scalar_where(cond.buffer, lhs.buffer, rhs))
    if isinstance(lhs, (int, float)) and isinstance(rhs, NDArray):
        assert cond.shape == rhs.shape
        return NDArray(lib.array_scalar_array_where(cond.buffer, lhs, rhs.buffer))
    raise NotImplementedError(f"not implemented for {type(cond)=} and {type(lhs)=}")


## -------------------------------------------------------------------------------------------------
## STACK/UNSTACK OPERATIONS
## -------------------------------------------------------------------------------------------------


def cat(arrays, dims: tp.Sequence[int]):
    assert isinstance(arrays, tp.Sequence)
    assert all(isinstance(array, NDArray) for array in arrays)
    assert isinstance(dims, tp.Sequence)
    assert all(array.ndim == arrays[0].ndim for array in arrays)
    ref = arrays[0].shape
    check_shape = lambda s: all(s[i] == ref[i] for i in range(len(s)) if i not in dims)

    for array in arrays:
        assert check_shape(array.shape)
        assert all(0 <= dim < array.shape[dim] for dim in dims)

    narray = len(arrays)
    ndim = len(dims)
    buffers = [array.buffer for array in arrays]
    return NDArray(lib.array_cat(buffers, narray, dims, ndim))


## -------------------------------------------------------------------------------------------------
## VIZ OPERATIONS
## -------------------------------------------------------------------------------------------------


def pprepr(array) -> str:
    assert isinstance(array, NDArray)
    return f"NDArray(f32[{','.join(map(str, array.shape))}])"


def ppstr(array) -> str:
    assert isinstance(array, NDArray)
    shape = array.shape
    out = StringIO()

    def recurse(index: list[int], depth: int):
        nonlocal out
        if depth == len(shape):
            out.write(f"{array[index]}")
            return

        out.write("[")
        for i in range(shape[depth]):
            recurse(index + [i], depth + 1)
            if i < shape[depth] - 1:
                out.write(", ")
        out.write("]")

    recurse([], 0)
    return out.getvalue()



## -------------------------------------------------------------------------------------------------
## OTHER OPERATIONS
## -------------------------------------------------------------------------------------------------


def as_strided(array, shape, stride):
    assert isinstance(array, NDArray)
    assert isinstance(shape, tp.Sequence)
    assert isinstance(stride, tp.Sequence)
    assert all(isinstance(i, int) for i in shape)
    assert all(isinstance(i, int) for i in stride)
    assert len(shape) == len(stride)
    shape = ffi.new("size_t[]", shape)
    stride = ffi.new("size_t[]", stride)
    return NDArray(lib.array_as_strided(array.buffer, shape, stride, len(shape)))