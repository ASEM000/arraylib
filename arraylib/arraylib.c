
#include "arraylib.h"

// ------------------------------------------------------------------
// MEMORY ALLOCATORS
// ------------------------------------------------------------------

void *alloc(size_t size) {
  assert(size > 0);
  void *ptr = malloc(size);
  assert(ptr != NULL);
  return ptr;
}

// ------------------------------------------------------------------
// UTILS
// ------------------------------------------------------------------

size_t prod(size_t *nums, size_t ndim) {
  size_t total = 1;
  for (size_t i = 0; i < ndim; i++)
    total *= nums[i];
  return total;
}

size_t *create_size_t(size_t ndim) {
  return (size_t *)alloc(sizeof(size_t) * ndim);
}

size_t *copy_size_t(size_t *dst, size_t *src, size_t size) {
  memcpy(dst, src, sizeof(size_t) * size);
  return dst;
}

size_t compute_flat_index(size_t *index, size_t *stride, size_t ndim) {
  if (ndim == 1)
    return index[0];

  size_t flat_index = 0;
  for (size_t i = 0; i < ndim; i++)
    flat_index += index[i] * stride[i];
  return flat_index;
}

size_t *compute_multi_index(size_t *dst, size_t flat_index, size_t *shape,
                            size_t ndim) {
  for (size_t i = 0; i < ndim; i++) {
    dst[ndim - i - 1] = flat_index % shape[ndim - i - 1];
    flat_index /= shape[ndim - i - 1];
  }
  return dst;
}

f32 clamp(f32 value, f32 minval, f32 maxval) {
  if (value < minval)
    return minval;
  if (value > maxval)
    return maxval;
  return value;
}

bool is_contiguous(NDArray *array) {
  size_t contiguous_stride = 1;
  for (ssize_t i = array->ndim - 1; i >= 0; i--) {
    if (contiguous_stride != array->stride[i])
      return false;
    contiguous_stride *= array->shape[i];
  }
  return true;
}

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

size_t *create_stride_from_shape(size_t *dst, size_t *shape, size_t ndim) {
  dst[ndim - 1] = 1;
  for (ssize_t i = ndim - 2; i >= 0; i--)
    dst[i] = dst[i + 1] * shape[i + 1];
  return dst;
}

// ------------------------------------------------------------------
// DATA
// ------------------------------------------------------------------

Data *data_empty(size_t size) {
  assert(size > 0);
  Data *data = (Data *)alloc(sizeof(Data));
  f32 *mem = (f32 *)alloc(sizeof(f32) * size);
  data->mem = mem;
  data->size = size;
  data->refs = 1;
  return data;
}

// ------------------------------------------------------------------
// ARRAY CREATION AND DESTRUCTION
// ------------------------------------------------------------------

NDArray *array_empty(size_t *shape, size_t ndim) {
  NDArray *array = (NDArray *)alloc(sizeof(NDArray));
  array->shape = copy_size_t(create_size_t(ndim), shape, ndim);
  array->stride = create_stride_from_shape(create_size_t(ndim), shape, ndim);
  array->data = data_empty(prod(shape, ndim));
  array->offset = 0;
  array->ndim = ndim;
  array->is_copy = false;
  return array;
}

void array_free(NDArray *array) {
  assert(array != NULL);
  if (--array->data->refs == 0) {
    free(array->data->mem);
    free(array->data);
  }
  free(array->shape);
  free(array->stride);
  free(array);
}

NDArray *array_shallow_copy(NDArray *array) {
  assert(array != NULL);
  NDArray *oarray = (NDArray *)alloc(sizeof(NDArray));
  oarray->ndim = array->ndim;
  size_t ndim = array->ndim;
  oarray->shape = copy_size_t(create_size_t(ndim), array->shape, array->ndim);
  oarray->stride = copy_size_t(create_size_t(ndim), array->stride, array->ndim);
  oarray->offset = array->offset;
  oarray->data = array->data;
  array->data->refs++;
  oarray->is_copy = true;
  return oarray;
}

NDArray *array_deep_copy(NDArray *array) {
  NDArray *oarray = array_empty(array->shape, array->ndim);
  size_t total = prod(array->shape, array->ndim);
  oarray->offset = array->offset;
  if (is_contiguous(array)) {
    memcpy(oarray->data->mem, array->data->mem, sizeof(f32) * total);
    return oarray;
  }

  size_t *index = create_size_t(array->ndim);
  memset(index, 0, array->ndim * sizeof(size_t));
  for (size_t flat_index = 0; flat_index < total; flat_index++) {
    index = compute_multi_index(index, flat_index, array->shape, array->ndim);
    f32 value = array_get_scalar_from_index(array, index);
    array_set_scalar_from_index(oarray, index, value);
  }
  free(index);
  return oarray;
}

NDArray *array_zeros(size_t *shape, size_t ndim) {
  NDArray *array = array_empty(shape, ndim);
  size_t total = prod(shape, ndim);
  memset(array->data->mem, 0, sizeof(f32) * total);
  return array;
}

NDArray *array_fill(f32 *elems, size_t *shape, size_t ndim) {
  size_t total = prod(shape, ndim);
  NDArray *array = array_empty(shape, ndim);
  for (size_t i = 0; i < total; i++)
    array->data->mem[i] = elems[i];
  return array;
}

NDArray *array_ones(size_t *shape, size_t ndim) {
  NDArray *array = array_empty(shape, ndim);
  size_t total = prod(shape, ndim);
  memset(array->data->mem, 1, sizeof(f32) * total);
  return array;
}

NDArray *array_arange(f32 start, f32 end, f32 step) {
  assert(start < end && start >= 0);
  size_t total = cdiv((end - start), step); // [start, end)
  NDArray *oarray = array_zeros((size_t[]){total}, 1);
  size_t running = start;
  for (size_t i = 0; i < total; i++) {
    oarray->data->mem[i] = running;
    running = add32(running, step);
  }
  return oarray;
}

NDArray *array_linspace(f32 start, f32 end, f32 n) {
  f32 dx = div32((end - start), n - 1);
  NDArray *oarray = array_arange(0, n, 1);
  for (size_t i = 0; i < oarray->data->size; i++)
    oarray->data->mem[i] = start + oarray->data->mem[i] * dx;
  return oarray;
}

// ------------------------------------------------------------------
// GETTERS/SETTERS
// ------------------------------------------------------------------

f32 array_get_scalar_from_index(NDArray *array, size_t *index) {
  size_t flat_index = compute_flat_index(index, array->stride, array->ndim);
  return array->data->mem[array->offset + flat_index];
}

NDArray *array_get_view_from_range(NDArray *array, size_t *start, size_t *end,
                                 size_t *step) {
  for (size_t i = 0; i < array->ndim; i++) {
    assert(start[i] < end[i] && start[i] >= 0 && end[i] <= array->shape[i]);
    assert(step[i] > 0);
  }
  size_t ndim = array->ndim;
  size_t *shape = copy_size_t(create_size_t(ndim), array->shape, ndim);
  size_t *stride = copy_size_t(create_size_t(ndim), array->stride, ndim);
  size_t offset = 0;
  for (size_t i = 0; i < array->ndim; i++) {
    shape[i] = ((end[i] - start[i]) / step[i]);
    stride[i] = array->stride[i] * step[i];
    offset += start[i] * array->stride[i];
  }
  assert(offset < array->data->size);
  NDArray *oarray = array_shallow_copy(array);
  free(oarray->shape);
  free(oarray->stride);
  oarray->shape = shape;
  oarray->stride = stride;
  oarray->offset = offset;
  return oarray;
}

NDArray *array_set_scalar_from_index(NDArray *array, size_t *index, f32 value) {
  size_t flat_index = compute_flat_index(index, array->stride, array->ndim);
  array->data->mem[array->offset + flat_index] = value;
  return array;
}

NDArray *array_set_view_from_range(NDArray *array, f32 value, size_t *start,
                                 size_t *end, size_t *step) {
  for (size_t i = 0; i < array->ndim; i++) {
    assert(start[i] < end[i] && start[i] >= 0 && end[i] <= array->shape[i]);
    assert(step[i] > 0);
  }
  size_t total = prod(array->shape, array->ndim);
  size_t *index = create_size_t(array->ndim);
  for (size_t i = 0; i < total; i++) {
    compute_multi_index(index, i, array->shape, array->ndim);
    array_set_scalar_from_index(array, index, value);
  }
  free(index);
  return array;
}

// ------------------------------------------------------------------
// RESHAPING
// ------------------------------------------------------------------

NDArray *array_reshape(NDArray *array, size_t *shape, size_t ndim) {
  assert(prod(shape, ndim) == array->data->size);
  NDArray *oarray = array_shallow_copy(array);
  free(oarray->shape);
  free(oarray->stride);
  oarray->shape = copy_size_t(create_size_t(ndim), shape, ndim);
  oarray->stride = create_stride_from_shape(create_size_t(ndim), shape, ndim);
  oarray->ndim = ndim;
  return oarray;
}

NDArray *array_expand_leading_axis(NDArray *array) {
  NDArray *oarray = array_shallow_copy(array);
  size_t ndim = array->ndim + 1;
  size_t *shape = create_size_t(ndim);
  size_t *stride = create_size_t(ndim);
  shape[0] = 1;
  stride[0] = 0;
  memcpy(shape + 1, array->shape, sizeof(size_t) * array->ndim);
  memcpy(stride + 1, array->stride, sizeof(size_t) * array->ndim);
  free(oarray->shape);
  free(oarray->stride);
  oarray->shape = shape;
  oarray->stride = stride;
  oarray->ndim = ndim;
  return oarray;
}

NDArray *array_transpose(NDArray *array, size_t *dst) {
  NDArray *oarray = array_shallow_copy(array);
  size_t ndim = array->ndim;
  size_t *shape = create_size_t(ndim), *stride = create_size_t(ndim);
  shape = copy_size_t(shape, array->shape, ndim);
  stride = copy_size_t(stride, array->stride, ndim);
  for (size_t i = 0; i < array->ndim; i++) {
    oarray->shape[i] = shape[dst[i]];
    oarray->stride[i] = stride[dst[i]];
  }
  free(shape);
  free(stride);
  return oarray;
}

NDArray *array_ravel(NDArray *array) {
  size_t total = prod(array->shape, array->ndim);
  NDArray *oarray = array_shallow_copy(array);
  free(oarray->shape);
  free(oarray->stride);
  oarray->shape = create_size_t(1);
  oarray->shape[0] = total;
  oarray->stride = create_size_t(1);
  oarray->stride[0] = 1;
  oarray->ndim = 1;
  return oarray;
}

// ------------------------------------------------------------------
// BINARY FUNCTIONS
// ------------------------------------------------------------------

f32 add32(f32 lhs, f32 rhs) { return lhs + rhs; }
f32 sub32(f32 lhs, f32 rhs) { return lhs - rhs; }
f32 mul32(f32 lhs, f32 rhs) { return lhs * rhs; }
f32 div32(f32 lhs, f32 rhs) { return rhs == 0 ? 0 : (f32)lhs / rhs; }
f32 pow32(f32 lhs, f32 rhs) { return powf(lhs, rhs); }

f32 eq32(f32 lhs, f32 rhs) { return lhs == rhs ? true : false; }
f32 neq32(f32 lhs, f32 rhs) { return lhs != rhs ? true : false; }
f32 geq32(f32 lhs, f32 rhs) { return lhs >= rhs ? true : false; }
f32 leq32(f32 lhs, f32 rhs) { return lhs <= rhs ? true : false; }
f32 gt32(f32 lhs, f32 rhs) { return lhs > rhs ? true : false; }
f32 lt32(f32 lhs, f32 rhs) { return lhs < rhs ? true : false; }

// ------------------------------------------------------------------
// ARRAY-SCALAR OPERATIONS
// ------------------------------------------------------------------

NDArray *array_scalar_op(NDArray *lhs, f32 rhs, binop fn) {
  NDArray *oarray = array_deep_copy(lhs);
  size_t total = prod(oarray->shape, oarray->ndim);
  for (size_t i = 0; i < total; i++)
    oarray->data->mem[i] = fn(oarray->data->mem[i], rhs);
  return oarray;
}

NDArray *array_scalar_add(NDArray *lhs, f32 rhs) {
  return array_scalar_op(lhs, rhs, add32);
}

NDArray *array_scalar_sub(NDArray *lhs, f32 rhs) {
  return array_scalar_op(lhs, rhs, sub32);
}

NDArray *array_scalar_mul(NDArray *lhs, f32 rhs) {
  return array_scalar_op(lhs, rhs, mul32);
}

NDArray *array_scalar_div(NDArray *lhs, f32 rhs) {
  return array_scalar_op(lhs, rhs, div32);
}

NDArray *array_scalar_pow(NDArray *lhs, f32 rhs) {
  return array_scalar_op(lhs, rhs, pow32);
}

// ------------------------------------------------------------------
// MATMUL
// ------------------------------------------------------------------

void block_matmul(NDArray *dst, NDArray *lhs, NDArray *rhs, size_t i0, size_t j0,
                  size_t k0, size_t imax, size_t jmax, size_t kmax) {
  for (size_t i = i0; i < imax; i++) {
    for (size_t j = j0; j < jmax; j++) {
      f32 sum = array_get_scalar_from_index(dst, (size_t[]){i, j});
      for (size_t k = k0; k < kmax; k++) {
        f32 lhs_val = array_get_scalar_from_index(lhs, (size_t[]){i, k});
        f32 rhs_val = array_get_scalar_from_index(rhs, (size_t[]){k, j});
        sum += lhs_val * rhs_val;
      }
      array_set_scalar_from_index(dst, (size_t[]){i, j}, sum);
    }
  }
}

NDArray *array_array_matmul(NDArray *lhs, NDArray *rhs) {
  // blocked matrix multiplication
  assert(lhs->ndim == 2 && rhs->ndim == 2);
  assert(lhs->shape[1] == rhs->shape[0]);
  size_t M = lhs->shape[0];
  size_t N = rhs->shape[1];
  size_t K = lhs->shape[1];
  NDArray *oarray = array_zeros((size_t[]){M, N}, 2);
  for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
    for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
      for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
        size_t imax = (i0 + BLOCK_SIZE < M) ? i0 + BLOCK_SIZE : M;
        size_t jmax = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
        size_t kmax = (k0 + BLOCK_SIZE < K) ? k0 + BLOCK_SIZE : K;
        block_matmul(oarray, lhs, rhs, i0, j0, k0, imax, jmax, kmax);
      }
    }
  }

  return oarray;
}

// ------------------------------------------------------------------
// ARRAY-ARRAY OPERATIONS
// ------------------------------------------------------------------

NDArray *array_array_scalar_op(NDArray *lhs, NDArray *rhs, binop fn) {
  assert(lhs->ndim == rhs->ndim);
  for (size_t i = 0; i < lhs->ndim; i++)
    assert(lhs->shape[i] == rhs->shape[i]);

  NDArray *oarray = array_empty(lhs->shape, lhs->ndim);
  size_t total = prod(lhs->shape, lhs->ndim);
  size_t *index = create_size_t(lhs->ndim);
  for (size_t i = 0; i < total; i++) {
    compute_multi_index(index, i, lhs->shape, lhs->ndim);
    f32 lval = array_get_scalar_from_index(lhs, index);
    f32 rval = array_get_scalar_from_index(rhs, index);
    array_set_scalar_from_index(oarray, index, fn(lval, rval));
  }
  free(index);
  return oarray;
}

NDArray *array_array_add(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, add32);
}

NDArray *array_array_sub(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, sub32);
}

NDArray *array_array_mul(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, mul32);
}

NDArray *array_array_div(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, div32);
}
NDArray *array_array_pow(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, pow32);
}

NDArray *array_array_dot(NDArray *lhs, NDArray *rhs) {
  assert(lhs->ndim == 1);
  assert(rhs->ndim == 1);
  assert(lhs->shape[0] == rhs->shape[0]);
  size_t index[1] = {0};
  f32 total = 0;
  for (size_t i = 0; i < lhs->shape[0]; index[0] = i++)
    total = add32(total, mul32(array_get_scalar_from_index(lhs, index),
                               array_get_scalar_from_index(rhs, index)));
  NDArray *oarray = array_empty((size_t[]){1}, 1);
  array_set_scalar_from_index(oarray, (size_t[]){0}, total);
  return oarray;
}

// comparison

NDArray *array_array_eq(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, eq32);
}

NDArray *array_array_neq(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, neq32);
}

NDArray *array_array_gt(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, gt32);
}

NDArray *array_array_geq(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, geq32);
}

NDArray *array_array_lt(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, lt32);
}

NDArray *array_array_leq(NDArray *lhs, NDArray *rhs) {
  return array_array_scalar_op(lhs, rhs, leq32);
}

// ------------------------------------------------------------------
// ELEMENTWISE OPERATIONS
// ------------------------------------------------------------------

f32 log32(f32 lhs) { return lhs == 0 ? 0 : logf(lhs); } // safe log
f32 neg32(f32 lhs) { return -lhs; }
f32 exp32(f32 lhs) { return expf(lhs); }

NDArray *array_op(NDArray *array, uniop fn) {
  NDArray *oarray = array_empty(array->shape, array->ndim);
  size_t total = prod(array->shape, array->ndim);
  size_t *index = create_size_t(array->ndim);
  for (size_t i = 0; i < total; i++) {
    compute_multi_index(index, i, array->shape, array->ndim);
    f32 lval = array_get_scalar_from_index(array, index);
    array_set_scalar_from_index(oarray, index, fn(lval));
  }
  free(index);
  return oarray;
}

NDArray *array_log(NDArray *array) { return array_op(array, log32); }
NDArray *array_neg(NDArray *array) { return array_op(array, neg32); }
NDArray *array_exp(NDArray *array) { return array_op(array, exp32); }