
#include "arraylib.h"

// -------------------------------------------------------------------------------------------------
// SIMLPLE ND-ARRAY IMPLMENTATION
// -------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------------
// MEMORY ALLOCATORS
// -------------------------------------------------------------------------------------------------

void* alloc(size_t size) {
    assert(size > 0 && "ValueError: negative size in malloc.");
    void* ptr = malloc(size);
    assert(ptr != NULL && "MemoryError: failed to allocate memory.");
    return ptr;
}

// -------------------------------------------------------------------------------------------------
// UTILS
// -------------------------------------------------------------------------------------------------

size_t prod(size_t* nums, size_t ndim) {
    assert(ndim > 0 && "ValueError: non-positive ndim");
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++)
        total *= nums[i];
    return total;
}

size_t* size_t_create(size_t ndim) {
    assert(ndim > 0 && "ValueError: non-positive ndim");
    return (size_t*)alloc(sizeof(size_t) * ndim);
}

size_t* size_t_set(size_t* dst, size_t value, size_t size) {
    assert(value >= 0 && "ValueError: negative size");
    for (size_t i = 0; i < size; i++)
        dst[i] = value;
    return dst;
}

size_t* size_t_copy(size_t* dst, size_t* src, size_t size) {
    assert(src != NULL && "ValueError: src copy is NULL");
    memcpy(dst, src, sizeof(size_t) * size);
    return dst;
}

size_t compute_flat_index(size_t* index, size_t* stride, size_t ndim) {
    size_t flat_index = 0;
    for (size_t i = 0; i < ndim; i++)
        flat_index += index[i] * stride[i];
    return flat_index;
}

f32 clamp(f32 value, f32 minval, f32 maxval) {
    if (value < minval)
        return minval;
    if (value > maxval)
        return maxval;
    return value;
}

bool is_contiguous(NDArray* array) {
    size_t contiguous_stride = 1;
    for (ssize_t i = array->ndim - 1; i >= 0; i--) {
        if (contiguous_stride != array->stride[i])
            return false;
        contiguous_stride *= array->shape[i];
    }
    return true;
}

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

size_t* compute_stride(size_t* dst, size_t* shape, size_t ndim) {
    dst[ndim - 1] = 1;
    for (ssize_t i = ndim - 2; i >= 0; i--)
        dst[i] = dst[i + 1] * shape[i + 1];
    return dst;
}

size_t* compute_bstride(size_t* dst, size_t* shape, size_t* stride, size_t ndim) {
    for (size_t i = 0; i < ndim; i++)
        dst[i] = (shape[i] - 1) * stride[i];
    return dst;
}

// -------------------------------------------------------------------------------------------------
// BINARY FUNCTIONS
// -------------------------------------------------------------------------------------------------

static inline f32 sum32(f32 lhs, f32 rhs) { return lhs + rhs; }
static inline f32 sub32(f32 lhs, f32 rhs) { return lhs - rhs; }
static inline f32 mul32(f32 lhs, f32 rhs) { return lhs * rhs; }
static inline f32 div32(f32 lhs, f32 rhs) { return (f32)lhs / rhs; }
static inline f32 pow32(f32 lhs, f32 rhs) { return powf(lhs, rhs); }

static inline f32 eq32(f32 lhs, f32 rhs) { return lhs == rhs ? true : false; }
static inline f32 neq32(f32 lhs, f32 rhs) { return lhs != rhs ? true : false; }
static inline f32 geq32(f32 lhs, f32 rhs) { return lhs >= rhs ? true : false; }
static inline f32 leq32(f32 lhs, f32 rhs) { return lhs <= rhs ? true : false; }
static inline f32 gt32(f32 lhs, f32 rhs) { return lhs > rhs ? true : false; }
static inline f32 lt32(f32 lhs, f32 rhs) { return lhs < rhs ? true : false; }

static inline f32 max32(f32 lhs, f32 rhs) { return lhs > rhs ? lhs : rhs; }
static inline f32 min32(f32 lhs, f32 rhs) { return lhs < rhs ? lhs : rhs; }

// -------------------------------------------------------------------------------------------------
// DATA
// -------------------------------------------------------------------------------------------------

Data* data_empty(size_t size) {
    assert(size > 0 && "ValueError: negative size in data_empty.");
    Data* data = (Data*)alloc(sizeof(Data));
    f32* mem = (f32*)alloc(sizeof(f32) * size);
    data->mem = mem;
    data->size = size;
    data->refs = 1;
    return data;
}

// -------------------------------------------------------------------------------------------------
// ARRAY CREATION AND DESTRUCTION
// -------------------------------------------------------------------------------------------------

NDArray* array_empty(size_t* shape, size_t ndim) {
    NDArray* array = (NDArray*)alloc(sizeof(NDArray));
    array->shape = size_t_copy(size_t_create(ndim), shape, ndim);
    array->stride = compute_stride(size_t_create(ndim), shape, ndim);
    array->bstride = compute_bstride(size_t_create(ndim), array->shape, array->stride, ndim);
    array->data = data_empty(prod(shape, ndim));
    array->offset = 0;
    array->ndim = ndim;
    array->view = false;
    return array;
}

void array_free(NDArray* array) {
    assert(array != NULL && "TypeError: array_free on NULL.");
    if (--array->data->refs == 0) {
        free(array->data->mem);
        free(array->data);
    }
    free(array->shape);
    free(array->stride);
    free(array);
}

// -------------------------------------------------------------------------------------------------
// ITERATOR
// -------------------------------------------------------------------------------------------------

NDIterator iter_create(
        f32* ptr,
        size_t* shape,
        size_t* stride,
        size_t* bstride,
        size_t* dims,
        size_t ndim) {
    for (size_t i = 0; i < ndim; i++)
        assert(dims[i] == ITERDIM || dims[i] < shape[i] && "ValueError: out of bounds.");

    NDIterator iter;
    size_t offset = 0;
    iter.shape = shape;                                        // does not own shape
    iter.stride = stride;                                      // does not own stride
    iter.bstride = bstride;                                    // does not own bstride
    iter.index = size_t_set(size_t_create(ndim), 0, ndim);     // own index
    iter.dims = size_t_copy(size_t_create(ndim), dims, ndim);  // own iterdims
    iter.counter = 0;
    iter.size = 1;
    for (size_t i = 0; i < ndim; i++)
        if (dims[i] == ITERDIM)
            iter.size *= shape[i];
        else
            offset += dims[i] * stride[i], iter.index[i] = dims[i];
    iter.ndim = ndim;
    iter.ptr = ptr + offset;
    return iter;
}

NDIterator iter_array(NDArray* array, size_t* iterdims) {
    return iter_create(
            &array->data->mem[array->offset],  // base pointer
            array->shape,                      // data shape
            array->stride,                     // data stride
            array->bstride,                    // data back stride
            iterdims,                          // dimensions to iter/fix
            array->ndim);
}

void iter_free(NDIterator* iterator) {
    free(iterator->index);
    free(iterator->dims);
}

bool iter_next(NDIterator* iter) {
    if (iter->counter >= iter->size)
        return false;

    if (iter->counter == 0) {
        iter->counter++;
        return true;
    }

    for (ssize_t i = iter->ndim - 1; i >= 0; i--) {
        if (iter->dims[i] != ITERDIM)
            continue;

        iter->index[i]++;
        if (iter->index[i] < iter->shape[i]) {
            iter->ptr += iter->stride[i];
            iter->counter++;
            return true;
        }
        iter->index[i] = 0;             // move to next
        iter->ptr -= iter->bstride[i];  // move to start of dim
    }
    iter->counter++;
    return true;
}

// -------------------------------------------------------------------------------------------------
// COPY
// -------------------------------------------------------------------------------------------------

NDArray* array_shallow_copy(NDArray* array) {
    assert(array != NULL && "TypeError: shallow copy of NULL.");
    NDArray* out_array = (NDArray*)alloc(sizeof(NDArray));
    out_array->ndim = array->ndim;
    size_t ndim = array->ndim;
    out_array->shape = size_t_copy(size_t_create(ndim), array->shape, ndim);
    out_array->stride = size_t_copy(size_t_create(ndim), array->stride, ndim);
    out_array->bstride = size_t_copy(size_t_create(ndim), array->bstride, ndim);
    out_array->offset = array->offset;
    out_array->data = array->data;
    array->data->refs++;
    out_array->view = true;
    return out_array;
}

NDArray* array_deep_copy(NDArray* array) {
    NDArray* out_array = array_empty(array->shape, array->ndim);
    size_t* dims = size_t_set(size_t_create(array->ndim), ITERDIM, array->ndim);
    NDIterator dst_iter = iter_array(out_array, dims);
    NDIterator src_iter = iter_array(array, dims);
    while (iter_next(&dst_iter) && iter_next(&src_iter))
        *dst_iter.ptr = *src_iter.ptr;
    iter_free(&dst_iter);
    iter_free(&src_iter);
    free(dims);
    return out_array;
}

// -------------------------------------------------------------------------------------------------
// INITIALIZATION
// -------------------------------------------------------------------------------------------------

NDArray* array_zeros(size_t* shape, size_t ndim) {
    NDArray* array = array_empty(shape, ndim);
    size_t total = prod(shape, ndim);
    memset(array->data->mem, 0, sizeof(f32) * total);
    return array;
}

NDArray* array_fill(f32* elems, size_t* shape, size_t ndim) {
    size_t total = prod(shape, ndim);
    NDArray* array = array_empty(shape, ndim);
    for (size_t i = 0; i < total; i++)
        array->data->mem[i] = elems[i];
    return array;
}

NDArray* array_ones(size_t* shape, size_t ndim) {
    NDArray* array = array_empty(shape, ndim);
    size_t total = prod(shape, ndim);
    for (size_t i = 0; i < total; i++)
        array->data->mem[i] = 1.0f;
    return array;
}

NDArray* array_arange(f32 start, f32 end, f32 step) {
    assert(start < end && start >= 0 && "ValueError: invalid array range.");
    size_t total = cdiv((end - start), step);  // [start, end)
    NDArray* out_array = array_zeros((size_t[]){total}, 1);
    size_t running = start;
    for (size_t i = 0; i < total; i++) {
        out_array->data->mem[i] = running;
        running = sum32(running, step);
    }
    return out_array;
}

NDArray* array_linspace(f32 start, f32 end, f32 n) {
    f32 dx = div32((end - start), n - 1);
    NDArray* out_array = array_arange(0, n, 1);
    for (size_t i = 0; i < out_array->data->size; i++)
        out_array->data->mem[i] = start + out_array->data->mem[i] * dx;
    return out_array;
}

// -------------------------------------------------------------------------------------------------
// GETTERS
// -------------------------------------------------------------------------------------------------

f32 array_get_scalar_from_index(NDArray* array, size_t* index) {
    size_t findex = compute_flat_index(index, array->stride, array->ndim);
    return array->data->mem[array->offset + findex];
}

NDArray* array_get_view_from_range(NDArray* array, size_t* start, size_t* end, size_t* step) {
    for (size_t i = 0; i < array->ndim; i++) {
        assert(start[i] < end[i] && start[i] >= 0 && "ValueError: invalid start.");
        assert(end[i] <= array->shape[i] && "ValueError: invald end.");
        assert(step[i] > 0 && "ValueError: negative step.");
    }
    size_t ndim = array->ndim;
    size_t* shape = size_t_copy(size_t_create(ndim), array->shape, ndim);
    size_t* stride = size_t_copy(size_t_create(ndim), array->stride, ndim);
    size_t offset = 0;
    for (size_t i = 0; i < array->ndim; i++) {
        shape[i] = cdiv(end[i] - start[i], step[i]);
        stride[i] = array->stride[i] * step[i];
        offset += start[i] * array->stride[i];
    }
    assert(offset < array->data->size && "ValueError: offset >= size.");
    NDArray* out_array = array_shallow_copy(array);
    free(out_array->shape);
    free(out_array->stride);
    out_array->shape = shape;
    out_array->stride = stride;
    out_array->offset = offset;
    return out_array;
}

// -------------------------------------------------------------------------------------------------
// SETTERS
// -------------------------------------------------------------------------------------------------

NDArray* array_set_scalar_from_index(NDArray* array, size_t* index, f32 value) {
    size_t findex = compute_flat_index(index, array->stride, array->ndim);
    array->data->mem[array->offset + findex] = value;
    return array;
}

NDArray* array_set_scalar_from_range(
        NDArray* array,
        size_t* start,
        size_t* end,
        size_t* step,
        f32 value) {
    for (size_t i = 0; i < array->ndim; i++) {
        assert(start[i] < end[i] && "ValueError: start index > end index.");
        assert(start[i] >= 0 && "ValueError: negative start index");
        assert(end[i] <= array->shape[i] && "ValueError: end index out of bounds.");
        assert(step[i] > 0 && "ValueError: negative step size.");
    }
    size_t ndim = array->ndim;
    size_t* view_shape = size_t_create(ndim);
    size_t* view_stride = size_t_create(ndim);
    size_t* view_bstride = size_t_create(ndim);
    for (size_t i = 0; i < ndim; i++) {
        view_shape[i] = cdiv(end[i] - start[i], step[i]);
        view_stride[i] = array->stride[i] * step[i];
        view_bstride[i] = (view_shape[i] - 1) * view_stride[i];
    }
    f32* ptr = &array->data->mem[array->offset];
    ptr += compute_flat_index(start, array->stride, array->ndim);
    size_t* iterdims = size_t_set(size_t_create(ndim), ITERDIM, ndim);
    NDIterator iter = iter_create(ptr, view_shape, view_stride, view_bstride, iterdims, ndim);
    while (iter_next(&iter))
        *iter.ptr = value;
    free(view_shape);
    free(view_stride);
    free(view_bstride);
    free(iterdims);
    iter_free(&iter);
    return array;
}

NDArray* array_set_view_from_array(
        NDArray* array,
        size_t* start,
        size_t* end,
        size_t* step,
        NDArray* value) {
    assert(array->ndim == value->ndim && "ValueError: dimension mismatch.");
    for (size_t i = 0; i < array->ndim; i++) {
        assert(start[i] < end[i] && "ValueError: start index > end index.");
        assert(start[i] >= 0 && "ValueError: negative start index");
        assert(end[i] <= array->shape[i] && "ValueError: end index out of bounds.");
        assert(step[i] > 0 && "ValueError: negative step size.");
    }
    size_t ndim = array->ndim;
    size_t* view_shape = size_t_create(ndim);
    size_t* view_stride = size_t_create(ndim);
    size_t* view_bstride = size_t_create(ndim);
    for (size_t i = 0; i < ndim; i++) {
        view_shape[i] = cdiv(end[i] - start[i], step[i]);
        view_stride[i] = array->stride[i] * step[i];
        view_bstride[i] = (view_shape[i] - 1) * view_stride[i];
    }
    assert((prod(view_shape, ndim) == prod(value->shape, ndim))
           && "ValueError: set value shape mismatch.");
    f32* ptr = &array->data->mem[array->offset];
    ptr += compute_flat_index(start, array->stride, array->ndim);
    size_t* iterdims = size_t_set(size_t_create(ndim), ITERDIM, ndim);
    NDIterator src = iter_create(ptr, view_shape, view_stride, view_bstride, iterdims, ndim);
    NDIterator dst = iter_array(value, iterdims);
    while (iter_next(&src) && iter_next(&dst))
        *src.ptr = *dst.ptr;
    iter_free(&src);
    iter_free(&dst);
    free(view_shape);
    free(view_stride);
    free(iterdims);
    return array;
}

// -------------------------------------------------------------------------------------------------
// RESHAPING
// -------------------------------------------------------------------------------------------------

NDArray* array_reshape(NDArray* array, size_t* shape, size_t ndim) {
    assert(prod(shape, ndim) == array->data->size);
    NDArray* out_array = is_contiguous(array) ? array_shallow_copy(array) : array_deep_copy(array);
    free(out_array->shape);
    free(out_array->stride);
    free(out_array->bstride);
    out_array->shape = size_t_copy(size_t_create(ndim), shape, ndim);
    out_array->stride = compute_stride(size_t_create(ndim), shape, ndim);
    out_array->bstride = compute_bstride(size_t_create(ndim), shape, out_array->stride, ndim);
    out_array->ndim = ndim;
    return out_array;
}

NDArray* array_transpose(NDArray* array, size_t* dst) {
    NDArray* out_array = is_contiguous(array) ? array_shallow_copy(array) : array_deep_copy(array);
    size_t ndim = array->ndim;
    size_t* shape = size_t_create(ndim);
    size_t* stride = size_t_create(ndim);
    size_t* bstride = size_t_create(ndim);
    shape = size_t_copy(shape, array->shape, ndim);
    stride = size_t_copy(stride, array->stride, ndim);
    bstride = size_t_copy(bstride, array->bstride, ndim);

    for (size_t i = 0; i < array->ndim; i++) {
        out_array->shape[i] = shape[dst[i]];
        out_array->stride[i] = stride[dst[i]];
        out_array->bstride[i] = bstride[dst[i]];
    }
    free(shape);
    free(stride);
    free(bstride);
    return out_array;
}

NDArray* array_move_axis(NDArray* array, size_t* src, size_t* dst, size_t ndim) {
    for (size_t i = 0; i < ndim; i++) {
        assert(src[i] >= 0 && src[i] < array->ndim && "ValueError: out of bounds");
        assert(dst[i] >= 0 && dst[i] < array->ndim && "ValueError: out of bounds");
    }

    size_t* bucket = size_t_set(size_t_create(array->ndim), ITERDIM, array->ndim);
    for (size_t i = 0; i < ndim; i++)
        bucket[src[i]] = 0;  // mark used axis
    NDArray* out_array = is_contiguous(array) ? array_shallow_copy(array) : array_deep_copy(array);
    size_t* swap_axes = size_t_set(size_t_create(array->ndim), ITERDIM, array->ndim);
    for (size_t i = 0; i < ndim; i++)
        swap_axes[dst[i]] = src[i];

    size_t j = 0;
    for (size_t i = 0; i < array->ndim; i++) {
        if (swap_axes[i] == ITERDIM) {  // free to fill
            while (bucket[j] != ITERDIM)
                j++;  // skip used axes
            swap_axes[i] = j++;
        }
    }

    for (size_t i = 0; i < array->ndim; i++) {
        out_array->shape[i] = array->shape[swap_axes[i]];
        out_array->stride[i] = array->stride[swap_axes[i]];
        out_array->bstride[i] = array->bstride[swap_axes[i]];
    }
    free(bucket);
    free(swap_axes);
    return out_array;
}

NDArray* array_ravel(NDArray* array) {
    size_t total = prod(array->shape, array->ndim);
    NDArray* out_array = is_contiguous(array) ? array_shallow_copy(array) : array_deep_copy(array);
    free(out_array->shape);
    free(out_array->stride);
    out_array->shape = size_t_create(1);
    out_array->shape[0] = total;
    out_array->stride = size_t_create(1);
    out_array->stride[0] = 1;
    out_array->ndim = 1;
    return out_array;
}

// -------------------------------------------------------------------------------------------------
// ARRAY-SCALAR OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_scalar_op(NDArray* array, f32 rhs, binop fn) {
    NDArray* out_array = array_empty(array->shape, array->ndim);
    size_t* dims = size_t_set(size_t_create(array->ndim), ITERDIM, array->ndim);
    NDIterator src = iter_array(array, dims);
    NDIterator dst = iter_array(out_array, dims);
    while (iter_next(&src) && iter_next(&dst))
        *dst.ptr = fn(*src.ptr, rhs);
    iter_free(&src);
    iter_free(&dst);
    free(dims);
    return out_array;
}

NDArray* array_scalar_add(NDArray* lhs, f32 rhs) { return array_scalar_op(lhs, rhs, sum32); }
NDArray* array_scalar_sub(NDArray* lhs, f32 rhs) { return array_scalar_op(lhs, rhs, sub32); }
NDArray* array_scalar_mul(NDArray* lhs, f32 rhs) { return array_scalar_op(lhs, rhs, mul32); }
NDArray* array_scalar_div(NDArray* lhs, f32 rhs) { return array_scalar_op(lhs, rhs, div32); }
NDArray* array_scalar_pow(NDArray* lhs, f32 rhs) { return array_scalar_op(lhs, rhs, pow32); }

// -------------------------------------------------------------------------------------------------
// MATMUL
// -------------------------------------------------------------------------------------------------

NDArray* array_array_matmul(NDArray* lhs, NDArray* rhs) {
    assert(lhs->ndim == 2 && rhs->ndim == 2);
    assert(lhs->shape[1] == rhs->shape[0]);
    size_t M = lhs->shape[0];
    size_t N = rhs->shape[1];
    size_t K = lhs->shape[1];

    NDArray* out_array = array_zeros((size_t[]){M, N}, 2);

    f32* lhs_p = &lhs->data->mem[lhs->offset];
    f32* rhs_p = &rhs->data->mem[rhs->offset];
    f32* out_p = &out_array->data->mem[out_array->offset];

    size_t lhs_s0 = lhs->stride[0];
    size_t lhs_s1 = lhs->stride[1];
    size_t rhs_s0 = rhs->stride[0];
    size_t rhs_s1 = rhs->stride[1];
    size_t out_s0 = out_array->stride[0];
    size_t out_s1 = out_array->stride[1];

    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                size_t imax = (i0 + BLOCK_SIZE < M) ? i0 + BLOCK_SIZE : M;
                size_t kmax = (k0 + BLOCK_SIZE < K) ? k0 + BLOCK_SIZE : K;
                size_t jmax = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
                for (size_t i = i0; i < imax; i++) {
                    for (size_t k = k0; k < kmax; k++) {
                        f32 lhs_val = lhs_p[i * lhs_s0 + k * lhs_s1];
                        for (size_t j = j0; j < jmax; j++) {
                            f32 rhs_val = rhs_p[k * rhs_s0 + j * rhs_s1];
                            f32 res_val = mul32(lhs_val, rhs_val);
                            f32 out_val = out_p[i * out_s0 + j * out_s1];
                            out_p[i * out_s0 + j * out_s1] = sum32(out_val, res_val);
                        }
                    }
                }
            }
        }
    }
    return out_array;
}

// -------------------------------------------------------------------------------------------------
// ARRAY-ARRAY OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_array_scalar_op(NDArray* lhs, NDArray* rhs, binop fn) {
    assert(lhs->ndim == rhs->ndim);
    for (size_t i = 0; i < lhs->ndim; i++)
        assert(lhs->shape[i] == rhs->shape[i]);
    NDArray* out = array_empty(lhs->shape, lhs->ndim);
    size_t* dims = size_t_set(size_t_create(lhs->ndim), ITERDIM, lhs->ndim);
    NDIterator lhs_iter = iter_array(lhs, dims);
    NDIterator rhs_iter = iter_array(rhs, dims);
    NDIterator out_iter = iter_array(out, dims);

    while (iter_next(&lhs_iter) && iter_next(&rhs_iter) && iter_next(&out_iter))
        *out_iter.ptr = fn(*lhs_iter.ptr, *rhs_iter.ptr);

    iter_free(&lhs_iter);
    iter_free(&rhs_iter);
    iter_free(&out_iter);
    free(dims);
    return out;
}

NDArray* array_array_sum(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, sum32);
}
NDArray* array_array_sub(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, sub32);
}
NDArray* array_array_mul(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, mul32);
}
NDArray* array_array_div(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, div32);
}
NDArray* array_array_pow(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, pow32);
}

// comparison
NDArray* array_array_eq(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, eq32);
}
NDArray* array_array_neq(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, neq32);
}
NDArray* array_array_gt(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, gt32);
}
NDArray* array_array_geq(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, geq32);
}
NDArray* array_array_lt(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, lt32);
}
NDArray* array_array_leq(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lhs, rhs, leq32);
}

// -------------------------------------------------------------------------------------------------
// ELEMENTWISE OPERATIONS
// -------------------------------------------------------------------------------------------------

static inline f32 log32(f32 lhs) { return lhs == 0 ? 0 : logf(lhs); }
static inline f32 neg32(f32 lhs) { return -lhs; }
static inline f32 exp32(f32 lhs) { return expf(lhs); }

NDArray* array_op(NDArray* array, uniop fn) {
    NDArray* out_array = array_empty(array->shape, array->ndim);
    size_t* dims = size_t_set(size_t_create(array->ndim), ITERDIM, array->ndim);
    NDIterator src = iter_array(array, dims);
    NDIterator dst = iter_array(out_array, dims);
    while (iter_next(&src) && iter_next(&dst))
        *dst.ptr = fn(*src.ptr);
    iter_free(&src);
    iter_free(&dst);
    free(dims);
    return out_array;
}

NDArray* array_log(NDArray* array) { return array_op(array, log32); }
NDArray* array_neg(NDArray* array) { return array_op(array, neg32); }
NDArray* array_exp(NDArray* array) { return array_op(array, exp32); }

// -------------------------------------------------------------------------------------------------
// REDUCTION OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_array_dot(NDArray* lhs, NDArray* rhs) {
    assert(lhs->ndim == 1 && "ValueError: lhs dimension != 1 for dot.");
    assert(rhs->ndim == 1 && "ValueError: rhs dimension != 1 for dot.");
    assert(lhs->shape[0] == rhs->shape[0] && "ValueError: dot shape mismatch.");
    size_t* dims = size_t_set(size_t_create(lhs->ndim), ITERDIM, lhs->ndim);  // iterate all
    f32 acc = 0.0f;
    NDIterator lhs_iter = iter_array(lhs, dims);
    NDIterator rhs_iter = iter_array(rhs, dims);
    while (iter_next(&lhs_iter) && iter_next(&rhs_iter))
        acc = sum32(acc, mul32(*lhs_iter.ptr, *rhs_iter.ptr));
    iter_free(&lhs_iter);
    iter_free(&rhs_iter);
    NDArray* out_array = array_empty((size_t[]){1}, 1);
    out_array->data->mem[0] = acc;
    free(dims);
    return out_array;
}

static NDArray* array_reshape_to_reduced_non_reduced(
        NDArray* array,
        size_t* reduce_dims,
        size_t reduce_ndim) {
    size_t* move_dst = size_t_create(reduce_ndim);
    for (size_t i = 0; i < reduce_ndim; i++)
        move_dst[i] = i;
    size_t ndim = array->ndim;
    NDArray* reshaped_array = array_move_axis(array, reduce_dims, move_dst, reduce_ndim);
    size_t total_size = prod(array->shape, ndim);
    size_t* reduce_shape = size_t_copy(size_t_create(ndim), array->shape, ndim);
    for (size_t i = 0; i < reduce_ndim; i++)
        reduce_shape[reduce_dims[i]] = 1;
    size_t rest_size = prod(reduce_shape, array->ndim);
    size_t reduce_size = total_size / rest_size;
    free(reduce_shape);
    free(move_dst);
    return array_reshape(reshaped_array, (size_t[]){reduce_size, rest_size}, 2);
}

NDArray* array_reduce(
        NDArray* array,
        size_t* reduce_dims,
        size_t reduce_ndim,
        binop acc_fn,
        f32 acc_init) {
    NDArray* reshaped_array = array_reshape_to_reduced_non_reduced(array, reduce_dims, reduce_ndim);
    size_t non_reduced_size = reshaped_array->shape[1];
    size_t* iterdims = size_t_set(size_t_create(2), ITERDIM, 2);
    NDArray* out_array = array_empty((size_t[]){non_reduced_size}, 1);
    NDIterator iter = iter_array(reshaped_array, iterdims);

    for (size_t i = 0; i < non_reduced_size; i++) {
        f32 sum = acc_init;
        iterdims[1] = i;                              // fix column i and iterate over rows
        iter = iter_array(reshaped_array, iterdims);  // reinitialize as iter size is from iterdims
        while (iter_next(&iter))
            sum = acc_fn(sum, *iter.ptr);  // sum all column i
        out_array->data->mem[i] = sum;
    }

    size_t ndim = array->ndim;
    size_t* out_shape = size_t_copy(size_t_create(ndim), array->shape, ndim);
    for (size_t i = 0; i < reduce_ndim; i++)
        out_shape[reduce_dims[i]] = 1;

    out_array = array_reshape(out_array, out_shape, array->ndim);
    iter_free(&iter);
    array_free(reshaped_array);
    free(out_shape);
    free(iterdims);
    return out_array;
}

NDArray* array_reduce_max(NDArray* array, size_t* reduce_dims, size_t ndim) {
    return array_reduce(array, reduce_dims, ndim, max32, -INFINITY);
}

NDArray* array_reduce_min(NDArray* array, size_t* reduce_dims, size_t ndim) {
    return array_reduce(array, reduce_dims, ndim, min32, INFINITY);
}

NDArray* array_reduce_sum(NDArray* array, size_t* reduce_dims, size_t ndim) {
    return array_reduce(array, reduce_dims, ndim, sum32, 0);
}