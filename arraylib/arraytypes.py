import arraylib
import functools as ft


def swop(f):
    @ft.wraps(f)
    def wrapper(a, b):
        return f(b, a)

    return wrapper


class NDArray:

    def __init__(self, buffer):
        self.buffer = buffer

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.buffer.shape[i] for i in range(self.ndim))

    @property
    def stride(self) -> tuple[int, ...]:
        return tuple(self.buffer.stride[i] for i in range(self.ndim))

    @property
    def bstride(self) -> tuple[int, ...]:
        return tuple(self.buffer.bstride[i] for i in range(self.ndim))
    
    @property
    def size(self) -> int:
        return self.buffer.size

    @property
    def view(self) -> bool:
        return self.buffer.view

    @property
    def offset(self) -> int:
        return self.buffer.offset

    @property
    def ndim(self) -> int:
        return self.buffer.ndim

    __del__ = arraylib.core.free
    __getitem__ = arraylib.core.getitem
    __setitem__ = arraylib.core.setitem
    __mul__ = arraylib.core.mul
    __rmul__ = swop(arraylib.core.mul)
    __neg__ = arraylib.core.neg
    __add__ = arraylib.core.add
    __radd__ = swop(arraylib.core.add)
    __sub__ = arraylib.core.sub
    __rsub__ = swop(arraylib.core.sub)
    __truediv__ = arraylib.core.div
    __rtruediv__ = swop(arraylib.core.div)
    __matmul__ = arraylib.core.matmul
    __str__ = arraylib.core.ppstr
    __repr__ = arraylib.core.pprepr
    __copy__ = arraylib.core.copy
    __matmul__ = arraylib.core.matmul
    __pow__ = arraylib.core.pow


class JVPNDArray(NDArray):
    def __init__(self, primal: NDArray, tangent: NDArray):
        assert primal.shape == tangent.shape
        assert primal.stride == tangent.stride
        self.primal = primal
        self.tangent = tangent