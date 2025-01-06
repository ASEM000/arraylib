import arraylib.primitive as primitive
from arraylib.impl import NDArray
import arraylib.core as core
from arraylib.arraytypes import JVPNDArray
import arraylib.core as core


@primitive.add_p.register(JVPNDArray, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs.primal, rhs.primal
    dx, dy = lhs.tangent, rhs.tangent
    return JVPNDArray(x + y, dx + dy)


@primitive.add_p.register(JVPNDArray, float)
@primitive.add_p.register(JVPNDArray, NDArray)
def _(lhs, rhs):
    x, y = lhs.primal, rhs
    dx = lhs.tangent
    return JVPNDArray(x + y, dx)


@primitive.add_p.register(float, JVPNDArray)
@primitive.add_p.register(NDArray, JVPNDArray)
def _(lhs, rhs):
    return core.add(rhs, lhs)


@primitive.sub_p.register(JVPNDArray, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs.primal, rhs.primal
    dx, dy = lhs.tangent, rhs.tangent
    return JVPNDArray(x - y, dx - dy)


@primitive.sub_p.register(JVPNDArray, float)
@primitive.sub_p.register(JVPNDArray, NDArray)
def _(lhs, rhs):
    x, y = lhs.primal, rhs
    dx = lhs.tangent
    return JVPNDArray(x - y, dx)


@primitive.sub_p.register(NDArray, JVPNDArray)
@primitive.sub_p.register(float, JVPNDArray)
def _(lhs, rhs):
    return -core.sub(rhs, lhs)


@primitive.mul_p.register(JVPNDArray, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs.primal, rhs.primal
    dx, dy = lhs.tangent, rhs.tangent
    primal = x * y
    tangent = x * dy + y * dx
    return JVPNDArray(primal, tangent)


@primitive.mul_p.register(JVPNDArray, float)
@primitive.mul_p.register(JVPNDArray, NDArray)
def _(lhs: NDArray, rhs):
    x, y = lhs.primal, rhs
    dx = lhs.tangent
    return JVPNDArray(x * y, dx * y)


@primitive.mul_p.register(NDArray, JVPNDArray)
@primitive.mul_p.register(float, JVPNDArray)
def _(lhs, rhs):
    return core.mul(rhs, lhs)


@primitive.div_p.register(JVPNDArray, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs.primal, rhs.primal
    dx, dy = lhs.tangent, rhs.tangent
    primal = x / y
    tangent = dx / y - x * dy / (y * y)
    return JVPNDArray(primal, tangent)


@primitive.div_p.register(JVPNDArray, float)
@primitive.div_p.register(JVPNDArray, NDArray)
def _(lhs: NDArray, rhs):
    x, y = lhs.primal, rhs
    dx = lhs.tangent
    return JVPNDArray(x / y, dx / y)


@primitive.div_p.register(NDArray, JVPNDArray)
@primitive.div_p.register(float, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs, rhs.primal
    dy = rhs.tangent
    return JVPNDArray(x / y, -x * dy / (y * y))


@primitive.pow_p.register(JVPNDArray, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs.primal, rhs.primal
    dx, dy = lhs.tangent, rhs.tangent
    primal = x**y
    tangent = y * x ** (y - 1.) * dx
    tangent += primal * core.log(x) * dy
    return JVPNDArray(primal, tangent)


@primitive.pow_p.register(JVPNDArray, float)
@primitive.pow_p.register(JVPNDArray, NDArray)
def _(lhs: NDArray, rhs):
    x, y = lhs.primal, rhs
    dx = lhs.tangent
    return JVPNDArray(x**y, y * x ** (y - 1) * dx)


@primitive.pow_p.register(NDArray, JVPNDArray)
@primitive.pow_p.register(float, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs, rhs.primal
    dy = rhs.tangent
    p = x**y
    return JVPNDArray(p, core.log(x) * p * dy)


@primitive.matmul_p.register(JVPNDArray, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs.primal, rhs.primal
    dx, dy = lhs.tangent, rhs.tangent
    primal = x @ y
    tangent = x @ dy + dx @ y
    return JVPNDArray(primal, tangent)


@primitive.matmul_p.register(JVPNDArray, NDArray)
def _(lhs: JVPNDArray, rhs):
    x, y = lhs.primal, rhs
    dx = lhs.tangent
    return JVPNDArray(x @ y, dx @ y)


@primitive.matmul_p.register(NDArray, JVPNDArray)
def _(lhs, rhs):
    x, y = lhs, rhs.primal
    dy = rhs.tangent
    return JVPNDArray(x @ y, x @ dy)


# reshape, transpose, ravel


@primitive.reshape_p.register(JVPNDArray)
def _(array, shape):
    primals = core.reshape(array.primal, shape)
    tangents = core.reshape(array.tangent, shape)
    return JVPNDArray(primals, tangents)


@primitive.transpose_p.register(JVPNDArray)
def _(array, dst):
    primals = core.transpose(array.primal, dst)
    tangents = core.transpose(array.tangent, dst)
    return JVPNDArray(primals, tangents)


@primitive.ravel_p.register(JVPNDArray)
def _(array):
    primals = core.ravel(array.primal)
    tangents = core.ravel(array.tangent)
    return JVPNDArray(primals, tangents)


@primitive.free_p.register(JVPNDArray)
def _(array):
    del array


@primitive.repr_p.register(JVPNDArray)
@primitive.str_p.register(JVPNDArray)
def _(array):
    return f"JVPNDArray(p={core.pprepr(array.primal)}, t={core.pprepr(array.tangent)})"


@primitive.exp_p.register(JVPNDArray)
def _(array):
    primal = core.exp(array.primal)
    tangent = array.tangent * primal
    return JVPNDArray(primal, tangent)
