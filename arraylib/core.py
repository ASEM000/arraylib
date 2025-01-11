"""User-facing Python API"""

from __future__ import annotations
import typing as tp
import functools as ft
from arraylib.clib import lib, ffi
import arraylib.primitive as primitive
import arraylib

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


def pprepr(array):
    return primitive.repr_p(array)


def ppstr(array):
    return primitive.str_p(array)


## -------------------------------------------------------------------------------------------------
## INITIALIZATION
## -------------------------------------------------------------------------------------------------


def array(elems: list[tp.Any]) -> arraylib.NDArray:
    assert isinstance(elems, tp.Sequence)
    shape = {}

    def flatten(elems, dim):
        if not isinstance(elems, tp.Sequence):
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
    return arraylib.NDArray(buffer=array)


def ones(shape: tp.Sequence[int]):
    assert isinstance(shape, tp.Sequence)
    assert all(isinstance(i, int) for i in shape)
    shape = tuple(shape)
    ndim = len(shape)
    shape = ffi.new("size_t[]", shape)
    return arraylib.NDArray(buffer=lib.array_ones(shape, ndim))


def zeros(shape: tp.Sequence[int]):
    assert isinstance(shape, tp.Sequence)
    assert all(isinstance(i, int) for i in shape)
    shape = tuple(shape)
    ndim = len(shape)
    shape = ffi.new("size_t[]", shape)
    return arraylib.NDArray(buffer=lib.array_zeros(shape, ndim))


def arange(start: int, end: int | None = None, step: int = 1):
    if end is None:
        end = start
        start = 0
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)
    assert (0 <= start < end) and (step > 0)
    return arraylib.NDArray(buffer=lib.array_arange(start, end, step))


def linspace(start: int, end: int, n: int):
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(n, int)
    assert (0 <= start < end) and (n > 0)
    return arraylib.NDArray(buffer=lib.array_linspace(start, end, n))


def free(array):
    primitive.free_p(array)


## -------------------------------------------------------------------------------------------------
## INTEROPERABILITY
## -------------------------------------------------------------------------------------------------


def to_buffer(array: arraylib.NDArray):
    array = array.buffer
    return ffi.buffer(array.data.mem, array.data.size * 4)


def from_buffer(buffer):
    size = len(buffer) // 4  # float size
    data = ffi.new("float[]", size)
    for i in range(size):
        data[i] = buffer[i]
    size = ffi.new("size_t[]", (size,))
    return arraylib.NDArray(buffer=lib.array_fill(data, size, 1))


def to_numpy(array: arraylib.NDArray):
    import numpy as np

    return np.ndarray(
        buffer=to_buffer(array),
        shape=array.shape,
        dtype=np.float32,
        strides=[s * 4 for s in array.stride],
    )


## -------------------------------------------------------------------------------------------------
## ARRAY-ARRAY/ARRAY-SCALAR OPERATIONS
## -------------------------------------------------------------------------------------------------


def add(lhs, rhs):
    """Add two arrays or an array and a scalar"""
    return primitive.add_p(lhs, rhs)


def sub(lhs, rhs):
    """Subtract two arrays or an array and a scalar"""
    return primitive.sub_p(lhs, rhs)


def mul(lhs, rhs):
    """Multiply two arrays or an array and a scalar"""
    return primitive.mul_p(lhs, rhs)


def div(lhs, rhs):
    """Divide two arrays or an array and a scalar"""
    return primitive.div_p(lhs, rhs)


def pow(lhs, rhs):
    """Raise an array to a power"""
    return primitive.pow_p(lhs, rhs)


def matmul(lhs, rhs):
    """Matrix multiply two arrays"""
    assert lhs.shape[-1] == rhs.shape[0]
    assert lhs.ndim == 2 and rhs.ndim == 2
    return primitive.matmul_p(lhs, rhs)


def dot(lhs, rhs):
    assert lhs.ndim == 1 and rhs.ndim == 1, "dot product requires 1D arrays"
    return primitive.dot_p(lhs, rhs)


## -------------------------------------------------------------------------------------------------
## ELEMENTWISE OPERATIONS
## -------------------------------------------------------------------------------------------------


def log(lhs):
    return primitive.log_p(lhs)


def neg(lhs):
    return primitive.neg_p(lhs)


def exp(lhs):
    return primitive.exp_p(lhs)


def wrap_elementwise_op(fn):
    # a function that accept a float and return a float
    # TODO: check the function signature
    # additionally check if the function is numpy ufunc
    @ffi.callback("float(float)")
    def wrapped(x):
        return fn(x)

    return wrapped


def apply(fn, array):
    """Apply an elementwise function to an array"""
    assert callable(fn)
    fn = wrap_elementwise_op(fn)
    return arraylib.NDArray(lib.array_op(fn, array.buffer))


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
    return primitive.reshape_p(array, shape)


def transpose(array, dst: tp.Sequence[int]):
    assert isinstance(dst, tp.Sequence)
    assert all(isinstance(i, int) for i in dst)
    dst = tuple(dst)
    assert len(dst) == array.ndim
    return primitive.transpose_p(array, dst)


def move_axis(array, src: tp.Sequence[int], dst: tp.Sequence[int]):
    assert isinstance(src, tp.Sequence)
    assert all(isinstance(i, int) for i in src)
    assert isinstance(dst, tp.Sequence)
    assert all(isinstance(i, int) for i in dst)
    src = tuple(src)
    dst = tuple(dst)
    assert len(src) == len(dst)
    assert 0 < len(src) <= array.ndim
    return primitive.move_axis_p(array, src, dst)


def ravel(array):
    return primitive.ravel_p(array)


## -------------------------------------------------------------------------------------------------
## COMPARISON OPERATIONS
## -------------------------------------------------------------------------------------------------


def eq(lhs, rhs):
    return primitive.eq_p(lhs, rhs)


def neq(lhs, rhs):
    return primitive.neq_p(lhs, rhs)


def leq(lhs, rhs):
    return primitive.leq_p(lhs, rhs)


def lt(lhs, rhs):
    return primitive.lt_p(lhs, rhs)


def geq(lhs, rhs):
    return primitive.geq_p(lhs, rhs)


def gt(lhs, rhs):
    return primitive.gt_p(lhs, rhs)


## -------------------------------------------------------------------------------------------------
## GETTERS
## -------------------------------------------------------------------------------------------------


def get_view_from_range(
    array,
    start: tuple[int, ...],
    end: tuple[int, ...],
    step: tuple[int, ...],
):
    assert isinstance(start, tuple)
    assert isinstance(end, tuple)
    assert isinstance(step, tuple)
    assert all(isinstance(i, int) for i in start)
    assert all(isinstance(i, int) for i in end)
    assert all(isinstance(i, int) for i in step)
    assert len(start) == len(end) == len(step) == array.ndim
    return primitive.get_view_from_range_p(array, start, end, step)


def get_scalar_from_index(array, index: tp.Sequence[int]) -> int:
    assert isinstance(index, tp.Sequence)
    assert all(isinstance(i, int) for i in index)
    assert len(index) == array.ndim
    assert all(idx < array.shape[i] for i, idx in enumerate(index))
    index = tuple(index)
    return primitive.get_scalar_from_index_p(array, index)


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


def set_scalar_from_index(array, index: tp.Sequence[int], value: int):
    assert isinstance(index, tp.Sequence)
    assert all(isinstance(i, int) for i in index)
    assert len(index) == array.ndim
    assert all(idx < array.shape[i] for i, idx in enumerate(index))
    index = tuple(index)
    return primitive.set_scalar_from_index_p(array, index, value)


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
    assert len(start) == len(end) == len(step) == array.ndim
    shape = array.shape
    assert all(0 <= si < ei <= shape[i] for i, (si, ei) in enumerate(zip(start, end)))
    return primitive.set_scalar_from_range_p(array, start, end, step, value)


def set_view_from_array(
    array,
    start: tuple[int, ...],
    end: tuple[int, ...],
    step: tuple[int, ...],
    value: arraylib.NDArray,
):
    assert isinstance(start, tuple)
    assert isinstance(end, tuple)
    assert isinstance(step, tuple)
    assert all(isinstance(i, int) for i in start)
    assert all(isinstance(i, int) for i in end)
    assert all(isinstance(i, int) for i in step)
    assert len(start) == len(end) == len(step) == value.ndim
    return primitive.set_view_from_array_p(array, start, end, step, value)


def setitem(array, index: int | slice | tp.Sequence[int | slice], value: float):
    index: tuple[tp.Any] = normalize_index(index)
    assert len(index) == array.ndim, "full index required to match ndim"
    if all(isinstance(i, int) for i in index):
        return set_scalar_from_index(array, index, value)
    if all(isinstance(i, slice) for i in index):
        start, end, step = slice_to_range(array.shape, array.ndim, index)
        if isinstance(value, arraylib.NDArray):
            return set_view_from_array(array, start, end, step, value)
        if isinstance(value, float):
            print(start, end, step, value)
            return primitive.set_scalar_from_range_p(array, start, end, step, value)
    return primitive.setitem_p(array, index, value)


## -------------------------------------------------------------------------------------------------
## COPY
## -------------------------------------------------------------------------------------------------


def copy(array, deep=False):
    array = array.buffer
    if deep:
        return arraylib.NDArray(lib.array_deep_copy(array))
    return lib.array_shallow_copy(array)


## -------------------------------------------------------------------------------------------------
## REDUCTIONS OPERATIONS
## -------------------------------------------------------------------------------------------------


def reduce_sum(array, axis=None):
    assert isinstance(axis, tp.Sequence) or axis is None
    axis = tuple(range(array.ndim)) if axis is None else tuple(axis)
    assert 0 <= len(axis) <= array.ndim, "axis out of bounds"
    return primitive.reduce_sum_p(array, axis) if len(axis) else array


def reduce_max(array, axis=None):
    assert isinstance(axis, tp.Sequence) or axis is None
    axis = tuple(range(array.ndim)) if axis is None else tuple(axis)
    assert 0 <= len(axis) <= array.ndim, "axis out of bounds"
    return primitive.reduce_max_p(array, axis) if len(axis) else array


def reduce_min(array, axis=None):
    assert isinstance(axis, tp.Sequence) or axis is None
    axis = tuple(range(array.ndim)) if axis is None else tuple(axis)
    assert 0 <= len(axis) <= array.ndim, "axis out of bounds"
    return primitive.reduce_min_p(array, axis) if len(axis) else array


def wrap_reduction_op(fn):
    # a function that accept a float and return a float
    # TODO: check the function signature

    @ffi.callback("float(float, float)")
    def wrapped(x, y):
        return fn(x, y)

    return wrapped


def reduce(fn, array, axis=None, init=0.0):
    """Reduce an array along an axis/axes with a python function"""
    assert isinstance(axis, tp.Sequence) or axis is None
    axis = tuple(range(array.ndim)) if axis is None else tuple(axis)
    assert 0 <= len(axis) <= array.ndim, "axis out of bounds"
    fn = wrap_reduction_op(fn)
    axis = ffi.new("size_t[]", axis)
    return arraylib.NDArray(lib.array_reduce(fn, array.buffer, axis, len(axis), init))
