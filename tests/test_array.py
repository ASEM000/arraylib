import pytest
import numpy as np
import arraylib as al
import numpy.testing as npt
import operator
from itertools import product


def assert_array_equal(al_array, np_array):
    np_array = np_array.astype(np.float32)
    npt.assert_array_equal(al.to_numpy(al_array), np_array.astype(np.float32))


def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result


def power_set(nums):
    if not nums:
        yield ()
        return
    num, *rest = nums
    for sub in power_set(rest):
        yield sub
        yield (num,) + sub


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [4, 5, 6]],
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        [[1]],
    ],
)
def test_array_creation(data):
    al_array = al.array(data)
    np_array = np.array(data)
    assert_array_equal(al_array, np_array)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (3, 2),
        (1, 1),
    ],
)
def test_ones(shape):
    al_array = al.ones(shape)
    np_array = np.ones(shape)
    assert_array_equal(al_array, np_array)


@pytest.mark.parametrize(
    "shape",
    [
        (2, 3),
        (3, 2),
        (1, 1),
    ],
)
def test_zeros(shape):
    al_array = al.zeros(shape)
    np_array = np.zeros(shape)
    assert_array_equal(al_array, np_array)


@pytest.mark.parametrize("start, stop, step", [(0, 5, 1), (1, 10, 2)])
def test_arange(start, stop, step):
    al_array = al.arange(start, stop, step)
    np_array = np.arange(start, stop, step)
    assert_array_equal(al_array, np_array)


@pytest.mark.parametrize("start, stop, num", [(0, 10, 5), (1, 100, 10)])
def test_linspace(start, stop, num):
    al_array = al.linspace(start, stop, num)
    np_array = np.linspace(start, stop, num)
    assert_array_equal(al_array, np_array)


@pytest.mark.parametrize(
    "a_data, b_data, op_func, op_name",
    [
        ([1, 2, 3], [4, 5, 6], operator.add, "add"),
        ([1, 2, 3], [4, 5, 6], operator.sub, "sub"),
        ([1, 2, 3], [4, 5, 6], operator.mul, "mul"),
        ([1, 2, 3], [4, 5, 6], operator.truediv, "div"),
    ],
)
def test_elementwise_ops(a_data, b_data, op_func, op_name):
    a_al = al.array(a_data)
    b_al = al.array(b_data)
    a_np = np.array(a_data, dtype=np.float32)
    b_np = np.array(b_data, dtype=np.float32)
    c_al = op_func(a_al, b_al)
    c_np = op_func(a_np, b_np)
    assert_array_equal(c_al, c_np)


@pytest.mark.parametrize(
    "a_data, b_data",
    [
        ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        ([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]]),
    ],
)
def test_matmul(a_data, b_data):
    a_al = al.array(a_data)
    b_al = al.array(b_data)
    c_al = a_al @ b_al

    a_np = np.array(a_data)
    b_np = np.array(b_data)
    c_np = a_np @ b_np

    assert_array_equal(c_al, c_np)


@pytest.mark.parametrize(
    "a_data, b_data",
    [([1, 2, 3], [4, 5, 6]), ([-1, 2, 3], [4, -5, 0])],
)
def test_dot(a_data, b_data):
    a_al = al.array(a_data)
    b_al = al.array(b_data)
    c_al = al.dot(a_al, b_al)

    a_np = np.array(a_data)
    b_np = np.array(b_data)
    c_np = np.dot(a_np, b_np).reshape(1)

    assert_array_equal(c_al, c_np)


@pytest.mark.parametrize(
    "data, new_shape",
    [
        ([[1, 2, 3], [4, 5, 6]], (3, 2)),
        ([[1, 2], [3, 4], [5, 6]], (2, 3)),
    ],
)
def test_reshape(data, new_shape):
    a_al = al.array(data)
    b_al = al.reshape(a_al, new_shape)

    a_np = np.array(data)
    b_np = np.reshape(a_np, new_shape)

    assert_array_equal(b_al, b_np)


@pytest.mark.parametrize(
    "data, axes",
    [
        ((3, 4, 5, 6), (1, 2, 0, 3)),  # 2D array
    ],
)
def test_transpose(data, axes):
    total = prod(data)
    a_al = al.reshape(al.arange(1, total + 1), data)
    b_al = al.transpose(a_al, axes)

    a_np = np.arange(1, total + 1).reshape(data)
    b_np = np.transpose(a_np, axes)
    assert_array_equal(b_al, b_np)


@pytest.mark.parametrize(
    "src, dst",
    [((1, 2), (2, 1)), ((3, 0), (1, 2))],
)
def test_move_axis(src, dst):
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    b_al = al.move_dim(a_al, src, dst)

    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    b_np = np.moveaxis(a_np, src, dst)
    assert_array_equal(b_al, b_np)


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [4, 5, 6]],
        [[1, 2], [3, 4], [5, 6]],
    ],
)
def test_ravel(data):
    a_al = al.array(data)
    b_al = al.ravel(a_al)

    a_np = np.array(data)
    b_np = np.ravel(a_np)

    assert_array_equal(b_al, b_np)


@pytest.mark.parametrize(
    "data, index, value",
    [
        ([[1, 2, 3], [4, 5, 6]], (0, 1), 10),
        ([[1, 2], [3, 4]], (1, 0), 5),
    ],
)
def test_getitem_setitem(data, index, value):
    a_al = al.array(data)
    assert a_al[index] == data[index[0]][index[1]]

    a_al[index] = value
    data[index[0]][index[1]] = value
    np_array = np.array(data)
    assert_array_equal(a_al, np_array)


@pytest.mark.parametrize("axis", tuple(power_set((0, 1, 2, 3))))
def test_reduce_sum(axis):
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    b_al = al.reduce_sum(a_al, dims=axis)

    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    b_np = np.sum(a_np, axis=axis, keepdims=True)
    assert_array_equal(b_al, b_np)


@pytest.mark.parametrize("axis", tuple(power_set((0, 1, 2, 3))))
def test_reduce_max(axis):
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    b_al = al.reduce_max(a_al, dims=axis)

    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    b_np = np.max(a_np, axis=axis, keepdims=True)

    assert_array_equal(b_al, b_np)


@pytest.mark.parametrize("axis", tuple(power_set((0, 1, 2, 3))))
def test_reduce_min(axis):
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    b_al = al.reduce_min(a_al, dims=axis)
    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    b_np = np.min(a_np, axis=axis, keepdims=True)
    assert_array_equal(b_al, b_np)


@pytest.mark.parametrize("index", product(range(3), range(4), range(5), range(6)))
def test_get_index_from_scalar(index):
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    assert a_al[index] == a_np[index]


@pytest.mark.parametrize("index", product(range(3), range(4), range(5), range(6)))
def test_set_index_from_scalar(index):
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    a_al[index] = 10
    a_np[index] = 10
    assert_array_equal(a_al, a_np)


@pytest.mark.parametrize(
    "fn",
    [np.log, np.exp, np.negative, np.sin, np.cos, np.tan, np.arctan, np.arcsin],
)
def test_unary_ops(fn):
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    b_al = al.apply(fn, a_al)
    b_np = fn(a_np)
    assert_array_equal(b_al, b_np)


def test_reduce():
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    reduce_fn = lambda x, y: x + y  # x is the accumulator, y is the current value
    b_al = al.reduce(reduce_fn, a_al, dims=(1, 2), init=0)
    b_np = a_np.sum(axis=(1, 2), keepdims=True)
    assert_array_equal(b_al, b_np)


def test_where():
    # array-array-array
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6)
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    cond = al.lt(a_al, 10.0)
    b_al = al.where(cond, a_al - 10.0, a_al + 10.0)
    b_np = np.where(a_np < 10.0, a_np - 10.0, a_np + 10.0)
    assert_array_equal(b_al, b_np)

    # array-array-scalar
    b_al = al.where(cond, a_al - 10.0, 10.0)
    b_np = np.where(a_np < 10.0, a_np - 10.0, 10.0)
    assert_array_equal(b_al, b_np)

    # array-scalar-array
    b_al = al.where(cond, 10.0, a_al - 10.0)
    b_np = np.where(a_np < 10.0, 10.0, a_np - 10.0)
    assert_array_equal(b_al, b_np)


@pytest.mark.parametrize(
    "lhs_shape, rhs_shape",
    [
        [(1, 2, 3), (1, 1, 2, 3)],
        [(1, 2, 3), (1, 2, 3)],
        [(1, 2, 3), (2, 3)],
        [(1, 2, 3), (3,)],
        [(1, 2, 3), (1, 3)],
        [(1, 2, 3), (1, 3)],
        [(1, 2, 3), (2, 1)],
        [(1, 2, 3), (1, 1)],
    ],
)
def test_broadcast(lhs_shape, rhs_shape):
    a_al = al.ones(lhs_shape) + al.ones(rhs_shape)
    a_np = np.ones(lhs_shape) + np.ones(rhs_shape)
    assert_array_equal(a_al, a_np)


def test_from_numpy():
    a_np = np.arange(1, 1 + 3 * 4 * 5 * 6).reshape(3, 4, 5, 6) / 10.0
    a_np = a_np.astype(np.float32)
    a_al = al.from_numpy(a_np)
    assert_array_equal(a_al, a_np)


def test_to_numpy():
    a_al = al.arange(1, 1 + 3 * 4 * 5 * 6) / 10.0
    a_al = al.reshape(a_al, (3, 4, 5, 6))
    a_np = al.to_numpy(a_al)
    assert_array_equal(a_al, a_np)


@pytest.mark.parametrize(
    "lhs, rhs, dim",
    [
        [(2, 4), (3, 4), 0],
        [(1, 6, 3), (1, 4, 3), 1],
        [(1, 2, 3), (1, 2, 4), 2],
    ],
)
def test_1d_cat(lhs, rhs, dim):
    a_al = al.arange(prod(lhs))
    a_al = al.reshape(a_al, lhs)
    b_al = al.arange(prod(rhs))
    b_al = al.reshape(b_al, rhs)
    c_al = al.cat([a_al, b_al], [dim])
    a_np = np.arange(prod(lhs)).reshape(lhs)
    b_np = np.arange(prod(rhs)).reshape(rhs)
    c_np = np.concatenate([a_np, b_np], axis=dim)
    assert_array_equal(c_al, c_np)


def test_nd_cat():
    # NOTE: nd cat is like numpy block but with zero padding
    a_al = al.arange(1, 1 + 1 * 3)
    a_al = al.reshape(a_al, (1, 3))
    b_al = al.arange(1, 1 + 2 * 2)
    b_al = al.reshape(b_al, (2, 2))

    a_np = np.arange(1, 1 + 1 * 3).reshape(1, 3)
    b_np = np.arange(1, 1 + 2 * 2).reshape(2, 2)
    c_np = np.block([[a_np, np.zeros((1, 2))], [np.zeros((2, 3)), b_np]])

    c_al = al.cat([a_al, b_al], [0, 1])

    assert_array_equal(c_al, c_np)
