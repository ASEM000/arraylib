import pytest
import numpy as np
import arraylib as al
import numpy.testing as npt
import operator


def assert_array_equal(al_array, np_array):
    np_array = np_array.astype(np.float32)
    npt.assert_array_equal(al.tonumpy(al_array), np_array.astype(np.float32))


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


@pytest.mark.parametrize(
    "start, stop, step",
    [(0, 5, 1), (1, 10, 2)],
)
def test_arange(start, stop, step):
    al_array = al.arange(start, stop, step)
    np_array = np.arange(start, stop, step)
    assert_array_equal(al_array, np_array)


@pytest.mark.parametrize(
    "start, stop, num",
    [(0, 10, 5), (1, 100, 10)],
)
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
    [([1, 2, 3], [4, 5, 6])],
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
        ([[1, 2, 3], [4, 5, 6]], (1, 0)),  # 2D array
        ([[1, 2], [3, 4], [5, 6]], (1, 0)),  # 3x2 array
    ],
)
def test_transpose(data, axes):
    a_al = al.array(data)
    b_al = al.transpose(a_al, axes)

    a_np = np.array(data)
    b_np = np.transpose(a_np, axes)

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