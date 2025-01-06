from __future__ import annotations
from io import StringIO
import arraylib.primitive as primitive
from arraylib.clib import lib, ffi
from arraylib import NDArray
from arraylib import core

# binary operations


@primitive.add_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_add(lhs.buffer, rhs.buffer))


@primitive.add_p.register(NDArray, float)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_scalar_add(lhs.buffer, rhs))


@primitive.add_p.register(float, NDArray)
def _(lhs, rhs) -> NDArray:
    return core.add(rhs, lhs)


@primitive.sub_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_sub(lhs.buffer, rhs.buffer))


@primitive.sub_p.register(NDArray, float)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_scalar_sub(lhs.buffer, rhs))


@primitive.sub_p.register(float, NDArray)
def _(lhs, rhs) -> NDArray:
    return -core.sub(rhs, lhs)


@primitive.mul_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_mul(lhs.buffer, rhs.buffer))


@primitive.mul_p.register(NDArray, float)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_scalar_mul(lhs.buffer, rhs))


@primitive.mul_p.register(float, NDArray)
def _(lhs, rhs) -> NDArray:
    return core.mul(rhs, lhs)


@primitive.div_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_div(lhs.buffer, rhs.buffer))


@primitive.pow_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_pow(lhs.buffer, rhs.buffer))


@primitive.pow_p.register(NDArray, float)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_scalar_pow(lhs.buffer, rhs))


@primitive.matmul_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_matmul(lhs.buffer, rhs.buffer))


@primitive.dot_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_dot(lhs.buffer, rhs.buffer))


# unary operations


@primitive.neg_p.register(NDArray)
def _(array) -> NDArray:
    return NDArray(buffer=lib.array_neg(array.buffer))


@primitive.log_p.register(NDArray)
def _(array) -> NDArray:
    return NDArray(buffer=lib.array_log(array.buffer))


@primitive.exp_p.register(NDArray)
def _(array) -> NDArray:
    return NDArray(buffer=lib.array_exp(array.buffer))


# reshaping


@primitive.reshape_p.register(NDArray)
def _(lhs, shape: tuple[int]) -> NDArray:
    ndim = len(shape)
    shape = ffi.new("size_t[]", shape)
    return NDArray(buffer=lib.array_reshape(lhs.buffer, shape, ndim))


@primitive.transpose_p.register(NDArray)
def _(lhs, dst: tuple[int, ...]) -> NDArray:
    dst = dst = ffi.new("size_t[]", dst)
    return NDArray(buffer=lib.array_transpose(lhs.buffer, dst))


@primitive.ravel_p.register(NDArray)
def _(array) -> NDArray:
    return NDArray(buffer=lib.array_ravel(array.buffer))


# comparison operations


@primitive.eq_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_eq(lhs.buffer, rhs.buffer))


@primitive.neq_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_neq(lhs.buffer, rhs.buffer))


@primitive.leq_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_leq(lhs.buffer, rhs.buffer))


@primitive.lt_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_lt(lhs.buffer, rhs.buffer))


@primitive.geq_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_geq(lhs.buffer, rhs.buffer))


@primitive.gt_p.register(NDArray, NDArray)
def _(lhs, rhs) -> NDArray:
    return NDArray(buffer=lib.array_array_gt(lhs.buffer, rhs.buffer))


# getter and setter


@primitive.get_view_from_range_p.register(NDArray)
def _(
    array,
    start: tuple[int, ...],
    end: tuple[int, ...],
    step: tuple[int, ...],
) -> NDArray:
    start = ffi.new("size_t[]", start)
    end = ffi.new("size_t[]", end)
    step = ffi.new("size_t[]", step)
    return NDArray(buffer=lib.array_get_view_from_range(array.buffer, start, end, step))


@primitive.get_scalar_from_index_p.register(NDArray)
def _(array, index: tuple[int]) -> float:
    index = ffi.new("size_t[]", index)
    return lib.array_get_scalar_from_index(array.buffer, index)


@primitive.set_view_from_range_p.register(NDArray)
def _(array, index: tuple[int, ...], value: int) -> NDArray:
    index = ffi.new("size_t[]", index)
    lib.array_set_view_from_index(array.buffer, index, value)
    return array


@primitive.set_scalar_from_index_p.register(NDArray)
def _(array, index: tuple[int, ...], value: int) -> NDArray:
    index = ffi.new("size_t[]", index)
    lib.array_set_scalar_from_index(array.buffer, index, value)
    return array


# repr and str


@primitive.repr_p.register(NDArray)
def _(array) -> str:
    return f"NDArray(f32[{','.join(map(str, array.shape))}])"


@primitive.str_p.register(NDArray)
def _(array) -> str:
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


@primitive.free_p.register(NDArray)
def _(array):
    lib.array_free(array.buffer)
    del array
