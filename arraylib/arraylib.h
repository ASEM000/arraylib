#ifndef ARRAYLIB_H
#define ARRAYLIB_H

#define BLOCK_SIZE 32

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// ------------------------------------------------------------------
// STRUCTS/TYPEDEFS
// ------------------------------------------------------------------

typedef float f32;
typedef f32 (*binop)(f32, f32);
typedef f32 (*uniop)(f32);

typedef struct {
  f32 *mem;
  size_t size;
  size_t refs;
} Data;

typedef struct {
  Data *data;
  size_t *shape;
  size_t *stride;
  size_t ndim;
  size_t offset;
  bool is_copy;
} NDArray;

// ------------------------------------------------------------------
// FUNCTION DECLARATIONS
// ------------------------------------------------------------------

// Memory allocators
void *alloc(size_t size);

// Utils
size_t prod(size_t *nums, size_t ndim);
size_t *create_size_t(size_t ndim);
size_t *copy_size_t(size_t *dst, size_t *src, size_t size);
size_t compute_flat_index(size_t *index, size_t *stride, size_t ndim);
size_t *compute_multi_index(size_t *dst, size_t flat_index,
                                   size_t *shape, size_t ndim);
bool is_contiguous(NDArray *array);
size_t cdiv(size_t a, size_t b);
size_t *create_stride_from_shape(size_t *dst, size_t *shape, size_t ndim);

// Data
Data *data_empty(size_t size);

f32 array_get_scalar_from_index(NDArray *array, size_t *index);
NDArray *array_get_view_from_range(NDArray *array, size_t *start, size_t *end,
                                 size_t *step);
NDArray *array_set_scalar_from_index(NDArray *array, size_t *index, f32 value);
NDArray *array_set_view_from_range(NDArray *array, f32 value, size_t *start,
                                 size_t *end, size_t *step);

// NDArray creation and destruction
NDArray *array_empty(size_t *shape, size_t ndim);
void array_free(NDArray *array);
NDArray *array_shallow_copy(NDArray *array);
NDArray *array_deep_copy(NDArray *array);
NDArray *array_zeros(size_t *shape, size_t ndim);
NDArray *array_fill(f32 *elems, size_t *shape, size_t ndim);
NDArray *array_ones(size_t *shape, size_t ndim);
NDArray *array_arange(f32 start, f32 end, f32 step);
NDArray *array_linspace(f32 start, f32 end, f32 n);

// Reshaping
NDArray *array_reshape(NDArray *array, size_t *shape, size_t ndim);
NDArray *array_expand_leading_axis(NDArray *array);
NDArray *array_transpose(NDArray *array, size_t *dst);
NDArray *array_ravel(NDArray *array);

// Binary functions
f32 add32(f32 lhs, f32 rhs);
f32 sub32(f32 lhs, f32 rhs);
f32 mul32(f32 lhs, f32 rhs);
f32 div32(f32 lhs, f32 rhs);
f32 eq32(f32 lhs, f32 rhs);
f32 neq32(f32 lhs, f32 rhs);
f32 geq32(f32 lhs, f32 rhs);
f32 leq32(f32 lhs, f32 rhs);
f32 gt32(f32 lhs, f32 rhs);
f32 lt32(f32 lhs, f32 rhs);

// NDArray-scalar operations
NDArray *array_scalar_op(NDArray *lhs, f32 rhs, binop fn);
NDArray *array_scalar_add(NDArray *lhs, f32 rhs);
NDArray *array_scalar_sub(NDArray *lhs, f32 rhs);
NDArray *array_scalar_mul(NDArray *lhs, f32 rhs);
NDArray *array_scalar_div(NDArray *lhs, f32 rhs);
NDArray *array_scalar_pow(NDArray *lhs, f32 rhs);

// NDArray-array operations
NDArray *array_array_matmul(NDArray *lhs, NDArray *rhs);
NDArray *array_array_scalar_op(NDArray *lhs, NDArray *rhs, binop fn);
NDArray *array_array_add(NDArray *lhs, NDArray *rhs);
NDArray *array_array_sub(NDArray *lhs, NDArray *rhs);
NDArray *array_array_mul(NDArray *lhs, NDArray *rhs);
NDArray *array_array_div(NDArray *lhs, NDArray *rhs);
NDArray *array_array_pow(NDArray *lhs, NDArray *rhs);
NDArray *array_array_dot(NDArray *lhs, NDArray *rhs);

// Comparison
NDArray *array_array_eq(NDArray *lhs, NDArray *rhs);
NDArray *array_array_neq(NDArray *lhs, NDArray *rhs);
NDArray *array_array_gt(NDArray *lhs, NDArray *rhs);
NDArray *array_array_geq(NDArray *lhs, NDArray *rhs);
NDArray *array_array_lt(NDArray *lhs, NDArray *rhs);
NDArray *array_array_leq(NDArray *lhs, NDArray *rhs);

// unary
NDArray *array_op(NDArray *array, uniop fn);
NDArray *array_log(NDArray *array);
NDArray *array_neg(NDArray *array);
NDArray *array_exp(NDArray *array);

#endif // ARRAYLIB_H