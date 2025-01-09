#ifndef ARRAYLIB_H
#define ARRAYLIB_H

#define BLOCK_SIZE 8
#define ITERDIM (size_t)(-1)

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

typedef struct
{
    f32* mem;
    size_t size;
    size_t refs;
} Data;

typedef struct
{
    Data* data;
    size_t* shape;
    size_t* stride;
    size_t ndim;
    size_t offset;
    bool view;
} NDArray;

typedef struct
{
    f32* ptr;
    size_t* shape;
    size_t* stride;
    size_t* bstride;
    size_t* index;
    size_t* dims;
    size_t counter;
    size_t size;
    size_t ndim;
} NDIterator;

// ------------------------------------------------------------------
// FUNCTION DECLARATIONS
// ------------------------------------------------------------------

// ------------------------------------------------------------------
// MEMORY ALLOCATORS
// ------------------------------------------------------------------

void* alloc(size_t size);

// ------------------------------------------------------------------
// UTILS
// ------------------------------------------------------------------

size_t prod(size_t* nums, size_t ndim);
size_t* size_t_create(size_t ndim);
size_t* size_t_set(size_t* dst, size_t value, size_t size);
size_t* size_t_copy(size_t* dst, size_t* src, size_t size);
size_t flat_index(size_t* index, size_t* stride, size_t ndim);
f32 clamp(f32 value, f32 minval, f32 maxval);
bool is_contiguous(NDArray* array);
size_t cdiv(size_t a, size_t b);
size_t* stride_from_shape(size_t* dst, size_t* shape, size_t ndim);
size_t* bstride_from_shape(size_t* dst, size_t* shape, size_t* stride, size_t ndim);

// ------------------------------------------------------------------
// DATA
// ------------------------------------------------------------------

Data* data_empty(size_t size);

// ------------------------------------------------------------------
// ARRAY CREATION AND DESTRUCTION
// ------------------------------------------------------------------

NDArray* array_empty(size_t* shape, size_t ndim);
void array_free(NDArray* array);

// ------------------------------------------------------------------
// ITERATOR
// ------------------------------------------------------------------

NDIterator iterator_create(f32* ptr, size_t* shape, size_t* stride, size_t* dims, size_t ndim);
NDIterator array_iter(NDArray* array);
NDIterator array_iter_axis(NDArray* array, size_t dim, size_t index);
void iterator_free(NDIterator* iterator);
bool iterator_iterate(NDIterator* iter);

// ------------------------------------------------------------------
// COPY
// ------------------------------------------------------------------

NDArray* array_shallow_copy(NDArray* array);
NDArray* array_deep_copy(NDArray* array);

// ------------------------------------------------------------------
// INITIALIZATION
// ------------------------------------------------------------------

NDArray* array_zeros(size_t* shape, size_t ndim);
NDArray* array_fill(f32* elems, size_t* shape, size_t ndim);
NDArray* array_ones(size_t* shape, size_t ndim);
NDArray* array_arange(f32 start, f32 end, f32 step);
NDArray* array_linspace(f32 start, f32 end, f32 n);

// ------------------------------------------------------------------
// GETTERS
// ------------------------------------------------------------------

f32 array_get_scalar_from_index(NDArray* array, size_t* index);
NDArray* array_get_view_from_range(NDArray* array, size_t* start, size_t* end, size_t* step);

// ------------------------------------------------------------------
// SETTERS
// ------------------------------------------------------------------

NDArray* array_set_scalar_from_index(NDArray* array, size_t* index, f32 value);
NDArray* array_set_scalar_from_range(NDArray* array, size_t* start, size_t* end, size_t* step, f32 value);
NDArray* array_set_view_from_array(NDArray* array, size_t* start, size_t* end, size_t* step, NDArray* value);

// ------------------------------------------------------------------
// RESHAPING
// ------------------------------------------------------------------

NDArray* array_reshape(NDArray* array, size_t* shape, size_t ndim);
NDArray* array_transpose(NDArray* array, size_t* dst);
NDArray* array_ravel(NDArray* array);

// ------------------------------------------------------------------
// ARRAY-SCALAR OPERATIONS
// ------------------------------------------------------------------

NDArray* array_scalar_op(NDArray* lhs, f32 rhs, binop fn);
NDArray* array_scalar_add(NDArray* lhs, f32 rhs);
NDArray* array_scalar_sub(NDArray* lhs, f32 rhs);
NDArray* array_scalar_mul(NDArray* lhs, f32 rhs);
NDArray* array_scalar_div(NDArray* lhs, f32 rhs);
NDArray* array_scalar_pow(NDArray* lhs, f32 rhs);

// ------------------------------------------------------------------
// MATMUL
// ------------------------------------------------------------------

NDArray* array_array_matmul(NDArray* lhs, NDArray* rhs);

// ------------------------------------------------------------------
// ARRAY-ARRAY OPERATIONS
// ------------------------------------------------------------------

NDArray* array_array_scalar_op(NDArray* lhs, NDArray* rhs, binop fn);
NDArray* array_array_add(NDArray* lhs, NDArray* rhs);
NDArray* array_array_sub(NDArray* lhs, NDArray* rhs);
NDArray* array_array_mul(NDArray* lhs, NDArray* rhs);
NDArray* array_array_div(NDArray* lhs, NDArray* rhs);
NDArray* array_array_pow(NDArray* lhs, NDArray* rhs);

// Comparison
NDArray* array_array_eq(NDArray* lhs, NDArray* rhs);
NDArray* array_array_neq(NDArray* lhs, NDArray* rhs);
NDArray* array_array_gt(NDArray* lhs, NDArray* rhs);
NDArray* array_array_geq(NDArray* lhs, NDArray* rhs);
NDArray* array_array_lt(NDArray* lhs, NDArray* rhs);
NDArray* array_array_leq(NDArray* lhs, NDArray* rhs);

// ------------------------------------------------------------------
// ELEMENTWISE OPERATIONS
// ------------------------------------------------------------------

NDArray* array_op(NDArray* array, uniop fn);
NDArray* array_log(NDArray* array);
NDArray* array_neg(NDArray* array);
NDArray* array_exp(NDArray* array);

// ------------------------------------------------------------------
// REDUCTION OPERATIONS
// ------------------------------------------------------------------
NDArray* array_array_dot(NDArray* lhs, NDArray* rhs);

#endif // ARRAYLIB_H