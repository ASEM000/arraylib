from arraylib.dispatch import bidispatch, unidispatch
import functools as ft

def _no_impl_error(*a, **k):
    raise NotImplementedError("This function is not implemented yet.")


# binary operations (8)
add_p = bidispatch(_no_impl_error)
sub_p = bidispatch(_no_impl_error)
mul_p = bidispatch(_no_impl_error)
div_p = bidispatch(_no_impl_error)
pow_p = bidispatch(_no_impl_error)
matmul_p = bidispatch(_no_impl_error)
dot_p = bidispatch(_no_impl_error)


# elementwise operations (3)
log_p = unidispatch(_no_impl_error)
neg_p = unidispatch(_no_impl_error)
exp_p = unidispatch(_no_impl_error)
apply_p = unidispatch(_no_impl_error)

# comparison operations (6)
eq_p = bidispatch(_no_impl_error)
neq_p = bidispatch(_no_impl_error)
leq_p = bidispatch(_no_impl_error)
lt_p = bidispatch(_no_impl_error)
geq_p = bidispatch(_no_impl_error)
gt_p = bidispatch(_no_impl_error)

# reshape (3)
ravel_p = unidispatch(_no_impl_error)
reshape_p = unidispatch(_no_impl_error)
transpose_p = unidispatch(_no_impl_error)
move_axis_p = unidispatch(_no_impl_error)

# setter and getter (4)
get_view_from_range_p = unidispatch(_no_impl_error)
get_scalar_from_index_p = unidispatch(_no_impl_error)
set_scalar_from_range_p = unidispatch(_no_impl_error)
set_scalar_from_index_p = unidispatch(_no_impl_error)
set_view_from_array_p = unidispatch(_no_impl_error)
# repr and str (2)
str_p = unidispatch(_no_impl_error)
repr_p = unidispatch(_no_impl_error)

# other
free_p = unidispatch(_no_impl_error)

# reductions
reduce_sum_p = unidispatch(_no_impl_error)
reduce_max_p = unidispatch(_no_impl_error)
reduce_min_p = unidispatch(_no_impl_error)
reduce_p = unidispatch(_no_impl_error)


# conditional

where_p = ft.partial(bidispatch, lhsnum=1, rhsnum=2)(_no_impl_error)