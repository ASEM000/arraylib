import pytest
import numpy as np
import arraylib


def assert_array_equal(custom_array, numpy_array):
    assert custom_array.shape == numpy_array.shape
    assert custom_array.tolist() == numpy_array.tolist()


def test_array_creation():
    data = [[1, 2, 3], [4, 5, 6]]
    custom_array = arraylib.array(data)
    numpy_array = np.array(data)
    assert_array_equal(custom_array, numpy_array)


def test_ones():
    shape = (2, 3)
    custom_array = arraylib.ones(shape)
    numpy_array = np.ones(shape)
    assert_array_equal(custom_array, numpy_array)


def test_zeros():
    shape = (2, 3)
    custom_array = arraylib.zeros(shape)
    numpy_array = np.zeros(shape)
    assert_array_equal(custom_array, numpy_array)


def test_arange():
    custom_array = arraylib.arange(5)
    numpy_array = np.arange(5)
    assert_array_equal(custom_array, numpy_array)


def test_linspace():
    custom_array = arraylib.linspace(0, 10, 5)
    numpy_array = np.linspace(0, 10, 5)
    assert_array_equal(custom_array, numpy_array)


def test_add():
    a_custom = arraylib.array([1, 2, 3])
    b_custom = arraylib.array([4, 5, 6])
    c_custom = a_custom + b_custom

    a_numpy = np.array([1, 2, 3])
    b_numpy = np.array([4, 5, 6])
    c_numpy = a_numpy + b_numpy

    assert_array_equal(c_custom, c_numpy)


def test_sub():
    a_custom = arraylib.array([1, 2, 3])
    b_custom = arraylib.array([4, 5, 6])
    c_custom = a_custom - b_custom

    a_numpy = np.array([1, 2, 3])
    b_numpy = np.array([4, 5, 6])
    c_numpy = a_numpy - b_numpy

    assert_array_equal(c_custom, c_numpy)


def test_mul():
    a_custom = arraylib.array([1, 2, 3])
    b_custom = arraylib.array([4, 5, 6])
    c_custom = a_custom * b_custom

    a_numpy = np.array([1, 2, 3])
    b_numpy = np.array([4, 5, 6])
    c_numpy = a_numpy * b_numpy

    assert_array_equal(c_custom, c_numpy)


def test_div():
    a_custom = arraylib.array([1, 2, 3])
    b_custom = arraylib.array([4, 5, 6])
    c_custom = a_custom / b_custom

    a_numpy = np.array([1, 2, 3])
    b_numpy = np.array([4, 5, 6])
    c_numpy = a_numpy / b_numpy

    assert_array_equal(c_custom, c_numpy)


def test_pow():
    a_custom = arraylib.array([1, 2, 3])
    b = 2
    c_custom = a_custom**b

    a_numpy = np.array([1, 2, 3])
    c_numpy = a_numpy**b

    assert_array_equal(c_custom, c_numpy)


def test_matmul():
    a_custom = arraylib.array([[1, 2], [3, 4]])
    b_custom = arraylib.array([[5, 6], [7, 8]])
    c_custom = a_custom @ b_custom

    a_numpy = np.array([[1, 2], [3, 4]])
    b_numpy = np.array([[5, 6], [7, 8]])
    c_numpy = a_numpy @ b_numpy

    assert_array_equal(c_custom, c_numpy)


def test_reshape():
    a_custom = arraylib.array([[1, 2, 3], [4, 5, 6]])
    b_custom = arraylib.reshape(a_custom, (3, 2))

    a_numpy = np.array([[1, 2, 3], [4, 5, 6]])
    b_numpy = np.reshape(a_numpy, (3, 2))

    assert_array_equal(b_custom, b_numpy)


def test_transpose():
    a_custom = arraylib.array([[1, 2, 3], [4, 5, 6]])
    b_custom = arraylib.transpose(a_custom, (1, 0))

    a_numpy = np.array([[1, 2, 3], [4, 5, 6]])
    b_numpy = np.transpose(a_numpy)

    assert_array_equal(b_custom, b_numpy)


def test_ravel():
    a_custom = arraylib.array([[1, 2, 3], [4, 5, 6]])
    b_custom = arraylib.ravel(a_custom)

    a_numpy = np.array([[1, 2, 3], [4, 5, 6]])
    b_numpy = np.ravel(a_numpy)

    assert_array_equal(b_custom, b_numpy)


def test_getitem():
    a_custom = arraylib.array([[1, 2, 3], [4, 5, 6]])
    assert a_custom[0, 1] == 2
    assert a_custom[1, 2] == 6


def test_setitem():
    a_custom = arraylib.array([[1, 2, 3], [4, 5, 6]])
    a_custom[0, 1] = 10
    assert a_custom.tolist() == [[1, 10, 3], [4, 5, 6]]