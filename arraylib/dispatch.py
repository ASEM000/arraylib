"""Dispatch by type key"""

import functools as ft
import types


def bidispatch(func, lhsnum: int = 0, rhsnum: int = 1):
    registry = {}

    def dispatch(lhstype, rhstype):
        return registry.get((lhstype, rhstype), func)

    def register(lhstype: type, rhstype: type):
        assert isinstance(lhstype, type)
        assert isinstance(rhstype, type)

        def decorator(impl):
            nonlocal lhstype, rhstype
            registry[(lhstype, rhstype)] = impl
            return impl

        return decorator

    @ft.wraps(func)
    def wrapper(*args):
        lhstyp = type(args[lhsnum])
        rhstyp = type(args[rhsnum])
        return dispatch(lhstyp, rhstyp)(*args)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = types.MappingProxyType(registry)
    return wrapper


def unidispatch(func, argnum: int = 0):

    registry = {}

    def dispatch(type):
        return registry.get(type, func)

    def register(klass):
        assert isinstance(klass, type)

        def decorator(impl):
            nonlocal klass
            registry[klass] = impl
            return impl

        return decorator

    @ft.wraps(func)
    def wrapper(*args):
        argtype = type(args[argnum])
        return dispatch(argtype)(*args)

    wrapper.register = register
    wrapper.dispatch = dispatch
    wrapper.registry = register
    return wrapper
