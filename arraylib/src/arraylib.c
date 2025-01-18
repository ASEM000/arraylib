
#include "arraylib.h"

// -------------------------------------------------------------------------------------------------
// SIMLPLE ND-ARRAY IMPLMENTATION
// -------------------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------------------
// MEMORY ALLOCATORS
// -------------------------------------------------------------------------------------------------

void* alloc(size_t size) {
    assert(size > 0 && "ValueError: non-positive size in malloc.");
    void* ptr = malloc(size);
    assert(ptr != NULL && "MemoryError: failed to allocate memory.");
    return ptr;
}

// -------------------------------------------------------------------------------------------------
// UTILS
// -------------------------------------------------------------------------------------------------

inline size_t prod(const size_t* nums, size_t ndim) {
    assert(ndim > 0 && "ValueError: non-positive ndim");
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++)
        total *= nums[i] == 0 ? 1 : nums[i];  // skip 0 shape
    return total;
}

inline size_t* size_t_alloc(size_t ndim) {
    assert(ndim > 0 && "ValueError: non-positive ndim");
    return (size_t*)alloc(sizeof(size_t) * ndim);
}

inline size_t* size_t_set(size_t* dst, size_t value, size_t size) {
    assert(value >= 0 && "ValueError: negative size");
    for (size_t i = 0; i < size; i++)
        dst[i] = value;
    return dst;
}

inline size_t* size_t_copy(size_t* dst, const size_t* src, size_t size) {
    assert(src != NULL && "ValueError: src copy is NULL");
    return memcpy(dst, src, sizeof(size_t) * size);
}

f32 clamp(f32 value, f32 minval, f32 maxval) {
    if (value < minval) return minval;
    if (value > maxval) return maxval;
    return value;
}

inline bool is_contiguous(const Layout* lay) {
    size_t contiguous_stride = 1;
    for (ssize_t i = lay->ndim - 1; i >= 0; i--) {
        if (contiguous_stride != lay->stride[i]) return false;
        contiguous_stride *= lay->shape[i];
    }
    return true;
}

inline size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

inline size_t* compute_stride(size_t* dst, const size_t* shape, const size_t ndim) {
    dst[ndim - 1] = 1;
    for (ssize_t i = ndim - 2; i >= 0; i--)
        dst[i] = dst[i + 1] * shape[i + 1];
    return dst;
}

inline size_t index_flatten(const size_t* index, const Layout* lay) {
    size_t flat_index = 0;
    for (size_t i = 0; i < lay->ndim; i++)
        flat_index += index[i] * lay->stride[i];
    return flat_index;
}

inline size_t* index_unflatten(size_t* dst, size_t index, const Layout* lay) {
    for (ssize_t i = lay->ndim - 1; i >= 0; i--)
        dst[i] = (index % lay->shape[i]), index /= lay->shape[i];
    return dst;
}

inline size_t offset_indexer(size_t index, const Layout* lay) {
    size_t offset = 0;
    for (ssize_t i = lay->ndim - 1; i >= 0; i--) {
        offset += (index % lay->shape[i]) * lay->stride[i];
        index /= lay->shape[i];
    }
    return offset;
}

// -------------------------------------------------------------------------------------------------
// DATA
// -------------------------------------------------------------------------------------------------

Data* data_empty(size_t size) {
    assert(size > 0 && "ValueError: non-postive size in data_empty.");
    Data* data = (Data*)alloc(sizeof(Data));
    f32* mem = (f32*)alloc(sizeof(f32) * size);
    data->mem = mem;
    data->size = size;
    data->refs = 1;
    return data;
}

// -------------------------------------------------------------------------------------------------
// LAYOUT
// -------------------------------------------------------------------------------------------------

Layout* layout_alloc(size_t ndim) {
    Layout* lay = (Layout*)alloc(sizeof(Layout));
    lay->ndim = ndim;
    lay->shape = size_t_alloc(ndim);
    lay->stride = size_t_alloc(ndim);
    return lay;
}

Layout* layout_copy(Layout* dst, const Layout* src) {
    dst->ndim = src->ndim;
    dst->shape = size_t_copy(dst->shape, src->shape, src->ndim);
    dst->stride = size_t_copy(dst->stride, src->stride, src->ndim);
    return dst;
}

void layout_free(Layout* lay) {
    free(lay->shape);
    free(lay->stride);
    free(lay);
}

bool is_same_shape(const Layout* lhs, const Layout* rhs) {
    if (lhs->ndim != rhs->ndim) return false;
    for (size_t i = 0; i < lhs->ndim; i++)
        if (lhs->shape[i] != rhs->shape[i]) return false;
    return false;
}

inline bool is_broadcastable(const Layout* lhs, const Layout* rhs) {
    ssize_t li = lhs->ndim;
    ssize_t ri = rhs->ndim;
    while (--li >= 0 && --ri >= 0)
        if ((lhs->shape[li] != rhs->shape[ri]) && lhs->shape[li] != 1 && rhs->shape[ri] != 1)
            return false;
    return true;
}

Layout** layout_broadcast(const Layout** lays, size_t nlay) {
    size_t max_ndim = lays[0]->ndim;
    for (size_t i = 1; i < nlay; i++)
        max_ndim = max_ndim < lays[i]->ndim ? lays[i]->ndim : max_ndim;

    Layout** blays = (Layout**)alloc(sizeof(Layout*) * nlay);
    for (size_t i = 0; i < nlay; i++)
        blays[i] = layout_alloc(max_ndim);

    for (size_t i = 0; i < max_ndim; i++) {
        size_t oi = (max_ndim - i - 1);
        size_t index_i[nlay];
        size_t shape_i[nlay];
        size_t max_shape_i = 0;
        for (size_t j = 0; j < nlay; j++) {
            index_i[j] = (i < lays[j]->ndim) ? (lays[j]->ndim - i - 1) : SIZE_MAX;
            shape_i[j] = (index_i[j] == SIZE_MAX) ? 1 : lays[j]->shape[index_i[j]];
            max_shape_i = max_shape_i < shape_i[j] ? shape_i[j] : max_shape_i;
        }
        for (size_t j = 0; j < nlay; j++) {
            blays[j]->shape[oi] = max_shape_i;
            bool is_broadcasted = (index_i[j] == SIZE_MAX || shape_i[j] != max_shape_i);
            blays[j]->stride[oi] = is_broadcasted ? 0 : lays[j]->stride[index_i[j]];
        }
    }
    return blays;
}

// -------------------------------------------------------------------------------------------------
// ARRAY CREATION AND DESTRUCTION
// -------------------------------------------------------------------------------------------------

NDArray* array_empty(const size_t* shape, size_t ndim) {
    NDArray* array = (NDArray*)alloc(sizeof(NDArray));
    array->lay = layout_alloc(ndim);  // contiguous lay
    array->lay->shape = size_t_copy(array->lay->shape, shape, ndim);
    array->lay->stride = compute_stride(array->lay->stride, shape, ndim);
    array->data = data_empty(prod(shape, ndim));
    array->ptr = array->data->mem;
    array->view = false;
    return array;
}

void array_free(NDArray* array) {
    assert(array != NULL && "TypeError: array_free on NULL.");
    if (--array->data->refs == 0) {
        free(array->data->mem);
        free(array->data);
    }
    layout_free(array->lay);
    free(array);
}

// -------------------------------------------------------------------------------------------------
// COPY
// -------------------------------------------------------------------------------------------------

NDArray* array_shallow_copy(const NDArray* src) {
    assert(src != NULL && "TypeError: shallow copy of NULL.");
    NDArray* dst = (NDArray*)alloc(sizeof(NDArray));
    dst->ptr = src->ptr;
    dst->lay = layout_copy(layout_alloc(src->lay->ndim), src->lay);
    dst->data = src->data;
    src->data->refs++;
    dst->view = true;
    return dst;
}

NDArray* array_deep_copy(const NDArray* src) {
    NDArray* dst = array_empty(src->lay->shape, src->lay->ndim);
    size_t size = prod(dst->lay->shape, dst->lay->ndim);

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
        dst->ptr[i] = src->ptr[offset_indexer(i, src->lay)];
    return dst;
}

// -------------------------------------------------------------------------------------------------
// INITIALIZATION
// -------------------------------------------------------------------------------------------------

NDArray* array_zeros(const size_t* shape, size_t ndim) {
    NDArray* array = array_empty(shape, ndim);
    size_t total = prod(shape, ndim);
    memset(array->data->mem, 0, sizeof(f32) * total);
    return array;
}

NDArray* array_fill(f32* elems, const size_t* shape, size_t ndim) {
    size_t total = prod(shape, ndim);
    NDArray* array = array_empty(shape, ndim);
    for (size_t i = 0; i < total; i++)
        array->data->mem[i] = elems[i];
    return array;
}

NDArray* array_ones(const size_t* shape, size_t ndim) {
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
        running = (running + step);
    }
    return out_array;
}

NDArray* array_linspace(f32 start, f32 end, f32 n) {
    f32 dx = (end - start) / (n - 1);
    NDArray* out_array = array_arange(0, n, 1);
    for (size_t i = 0; i < out_array->data->size; i++)
        out_array->ptr[i] = start + out_array->ptr[i] * dx;
    return out_array;
}

// -------------------------------------------------------------------------------------------------
// GETTERS
// -------------------------------------------------------------------------------------------------

f32 array_get_scalar_from_index(const NDArray* array, const size_t* index) {
    return array->ptr[offset_indexer(index_flatten(index, array->lay), array->lay)];
}

NDArray* array_get_view_from_range(
        const NDArray* src,
        const size_t* start,
        const size_t* end,
        const size_t* step) {
    for (size_t i = 0; i < src->lay->ndim; i++) {
        assert(start[i] < end[i] && start[i] >= 0 && "ValueError: invalid start.");
        assert(end[i] <= src->lay->shape[i] && "ValueError: invald end.");
        assert(step[i] > 0 && "ValueError: negative step.");
    }
    Layout* vlay = layout_alloc(src->lay->ndim);
    size_t offset = 0;
    for (size_t i = 0; i < src->lay->ndim; i++) {
        vlay->shape[i] = cdiv(end[i] - start[i], step[i]);
        vlay->stride[i] = src->lay->stride[i] * step[i];
        offset += start[i] * src->lay->stride[i];
    }
    assert(offset < src->data->size && "ValueError: offset >= size.");
    NDArray* dst = array_shallow_copy(src);
    FREE(dst->lay);
    dst->lay = vlay;
    dst->ptr = src->ptr + offset;
    return dst;
}

// -------------------------------------------------------------------------------------------------
// SETTERS
// -------------------------------------------------------------------------------------------------

NDArray* array_set_scalar_from_index(NDArray* array, f32 value, const size_t* index) {
    array->ptr[index_flatten(index, array->lay)] = value;
    return array;
}
NDArray* array_set_scalar_from_range(
        NDArray* dst,
        f32 value,
        const size_t* start,
        const size_t* end,
        const size_t* step) {
    for (size_t i = 0; i < dst->lay->ndim; i++) {
        assert(start[i] < end[i] && "ValueError: start index > end index.");
        assert(start[i] >= 0 && "ValueError: negative start index");
        assert(end[i] <= dst->lay->shape[i] && "ValueError: end index out of bounds.");
        assert(step[i] > 0 && "ValueError: negative step size.");
    }
    size_t ndim = dst->lay->ndim;
    Layout* vlay = layout_alloc(dst->lay->ndim);
    for (size_t i = 0; i < ndim; i++) {
        vlay->shape[i] = cdiv(end[i] - start[i], step[i]);
        vlay->shape[i] = dst->lay->stride[i] * step[i];
    }
    f32* ptr = dst->ptr + index_flatten(start, dst->lay);
    size_t size = prod(vlay->shape, vlay->ndim);
#pragma omp parallel
    for (size_t i = 0; i < size; i++)
        ptr[offset_indexer(i, dst->lay)] = value;
    return dst;
}

NDArray* array_set_view_from_array(
        NDArray* dst,
        const NDArray* src,
        const size_t* start,
        const size_t* end,
        const size_t* step) {
    assert(dst->lay->ndim == src->lay->ndim && "ValueError: dimension mismatch.");
    size_t ndim = dst->lay->ndim;
    for (size_t i = 0; i < ndim; i++) {
        assert(start[i] < end[i] && "ValueError: start index > end index.");
        assert(start[i] >= 0 && "ValueError: negative start index");
        assert(end[i] <= dst->lay->shape[i] && "ValueError: end index out of bounds.");
        assert(step[i] > 0 && "ValueError: negative step size.");
    }
    Layout* vlay = layout_alloc(ndim);
    vlay->ndim = ndim;
    for (size_t i = 0; i < ndim; i++) {
        vlay->shape[i] = cdiv(end[i] - start[i], step[i]);
        vlay->stride[i] = dst->lay->stride[i] * step[i];
    }
    size_t size = prod(vlay->shape, ndim);
    assert((size == prod(src->lay->shape, ndim)) && "ValueError: shape mismatch.");
    f32* pdst = dst->ptr + index_flatten(start, dst->lay);
#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
        pdst[offset_indexer(i, vlay)] = src->ptr[offset_indexer(i, src->lay)];
    return dst;
}

// -------------------------------------------------------------------------------------------------
// RESHAPING
// -------------------------------------------------------------------------------------------------

NDArray* array_reshape(const NDArray* src, const size_t* shape, size_t ndim) {
    assert(prod(shape, ndim) == src->data->size);
    NDArray* dst = is_contiguous(src->lay) ? array_shallow_copy(src) : array_deep_copy(src);
    FREE(dst->lay);
    dst->lay = layout_alloc(ndim);
    dst->lay->shape = size_t_copy(dst->lay->shape, shape, ndim);
    dst->lay->stride = compute_stride(dst->lay->stride, shape, ndim);
    dst->lay->ndim = ndim;
    return dst;
}

NDArray* array_transpose(const NDArray* src, const size_t* dims) {
    NDArray* dst = is_contiguous(src->lay) ? array_shallow_copy(src) : array_deep_copy(src);
    size_t ndim = src->lay->ndim;
    Layout* original_layout = layout_copy(layout_alloc(src->lay->ndim), src->lay);
    for (size_t i = 0; i < ndim; i++) {
        dst->lay->shape[i] = original_layout->shape[dims[i]];
        dst->lay->stride[i] = original_layout->stride[dims[i]];
    }
    FREE(original_layout);
    return dst;
}

NDArray* array_move_dim(const NDArray* src, const size_t* from, const size_t* to, size_t ndim) {
    for (size_t i = 0; i < ndim; i++) {
        assert(from[i] >= 0 && from[i] < src->lay->ndim && "ValueError: out of bounds");
        assert(from[i] >= 0 && from[i] < src->lay->ndim && "ValueError: out of bounds");
    }

    char bucket[src->lay->ndim];
    memset(bucket, 0, src->lay->ndim);

    for (size_t i = 0; i < ndim; i++)
        bucket[from[i]] = 1;  // mark used dimension
    NDArray* dst = is_contiguous(src->lay) ? array_shallow_copy(src) : array_deep_copy(src);
    size_t to_from[src->lay->ndim];
    size_t_set(to_from, ITERDIM, src->lay->ndim);
    for (size_t i = 0; i < ndim; i++)
        to_from[to[i]] = from[i];

    size_t j = 0;
    for (size_t i = 0; i < src->lay->ndim; i++) {
        if (to_from[i] == ITERDIM) {  // free to fill
            while (j < src->lay->ndim && bucket[j] != 0)
                j++;  // skip used axes
            to_from[i] = j++;
        }
    }

    for (size_t i = 0; i < src->lay->ndim; i++) {
        dst->lay->shape[i] = src->lay->shape[to_from[i]];
        dst->lay->stride[i] = src->lay->stride[to_from[i]];
    }
    return dst;
}

NDArray* array_ravel(const NDArray* src) {
    size_t total = prod(src->lay->shape, src->lay->ndim);
    NDArray* dst = is_contiguous(src->lay) ? array_shallow_copy(src) : array_deep_copy(src);
    FREE(dst->lay);
    dst->lay = layout_alloc(1);
    dst->lay->shape = size_t_set(size_t_alloc(1), total, 1);
    dst->lay->stride = size_t_set(size_t_alloc(1), 1, 1);
    return dst;
}

// -------------------------------------------------------------------------------------------------
// ARRAY-SCALAR OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_scalar_op(binop fn, const NDArray* src, f32 rhs) {
    NDArray* dst = array_empty(src->lay->shape, src->lay->ndim);
    size_t size = prod(src->lay->shape, src->lay->ndim);
    // #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        size_t doffset = offset_indexer(i, dst->lay);
        size_t soffset = offset_indexer(i, src->lay);
        dst->ptr[doffset] = fn(src->ptr[soffset], rhs);
    }
    return dst;
}

// -------------------------------------------------------------------------------------------------
// MATMUL
// -------------------------------------------------------------------------------------------------

NDArray* array_array_matmul(const NDArray* lhs, const NDArray* rhs) {
    assert(lhs->lay->ndim == 2 && rhs->lay->ndim == 2);
    assert(lhs->lay->shape[1] == rhs->lay->shape[0]);
    size_t M = lhs->lay->shape[0];
    size_t N = rhs->lay->shape[1];
    size_t K = lhs->lay->shape[1];

    NDArray* out = array_zeros((size_t[]){M, N}, 2);

    f32* plhs = lhs->ptr;
    f32* prhs = rhs->ptr;
    f32* pout = out->ptr;

    size_t lhs_s0 = lhs->lay->stride[0];
    size_t lhs_s1 = lhs->lay->stride[1];
    size_t rhs_s0 = rhs->lay->stride[0];
    size_t rhs_s1 = rhs->lay->stride[1];
    size_t out_s0 = out->lay->stride[0];
    size_t out_s1 = out->lay->stride[1];

#pragma omp parallel for collapse(2)
    for (size_t i0 = 0; i0 < M; i0 += BLOCK_SIZE) {
        for (size_t k0 = 0; k0 < K; k0 += BLOCK_SIZE) {
            for (size_t j0 = 0; j0 < N; j0 += BLOCK_SIZE) {
                size_t imax = (i0 + BLOCK_SIZE < M) ? i0 + BLOCK_SIZE : M;
                size_t kmax = (k0 + BLOCK_SIZE < K) ? k0 + BLOCK_SIZE : K;
                size_t jmax = (j0 + BLOCK_SIZE < N) ? j0 + BLOCK_SIZE : N;
                for (size_t i = i0; i < imax; i++) {
                    for (size_t k = k0; k < kmax; k++) {
                        f32 lhs_val = plhs[i * lhs_s0 + k * lhs_s1];
                        for (size_t j = j0; j < jmax; j++) {
                            f32 rhs_val = prhs[k * rhs_s0 + j * rhs_s1];
                            f32 res_val = (lhs_val * rhs_val);
                            f32 out_val = pout[i * out_s0 + j * out_s1];
                            pout[i * out_s0 + j * out_s1] = (out_val + res_val);
                        }
                    }
                }
            }
        }
    }
    return out;
}
// -------------------------------------------------------------------------------------------------
// ARRAY-ARRAY OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_array_scalar_op(binop fn, const NDArray* lhs, const NDArray* rhs) {
    assert(is_broadcastable(lhs->lay, rhs->lay) && "ValueError: can not broadcast.");
    const Layout* lays[2] = {lhs->lay, rhs->lay};
    Layout** blays = layout_broadcast(lays, 2);
    Layout* lhs_blay = blays[0];
    Layout* rhs_blay = blays[1];
    size_t size = prod(lhs_blay->shape, lhs_blay->ndim);
    NDArray* out = array_empty(lhs_blay->shape, lhs_blay->ndim);

#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        f32 lval = lhs->ptr[offset_indexer(i, lhs_blay)];
        f32 rval = rhs->ptr[offset_indexer(i, rhs_blay)];
        out->ptr[i] = fn(lval, rval);
    }

    FREE(lhs_blay);
    FREE(rhs_blay);
    FREE(blays);
    return out;
}

// -------------------------------------------------------------------------------------------------
// ELEMENTWISE OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_op(uniop fn, const NDArray* src) {
    NDArray* dst = array_empty(src->lay->shape, src->lay->ndim);
    size_t size = prod(src->lay->shape, src->lay->ndim);
#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
        dst->ptr[i] = fn(src->ptr[offset_indexer(i, src->lay)]);
    return dst;
}

// -------------------------------------------------------------------------------------------------
// REDUCTION OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_array_dot(const NDArray* lhs, const NDArray* rhs) {
    assert(lhs->lay->ndim == 1 && "ValueError: lhs dimension != 1 for dot.");
    assert(rhs->lay->ndim == 1 && "ValueError: rhs dimension != 1 for dot.");
    assert(lhs->lay->shape[0] == rhs->lay->shape[0] && "ValueError: dot shape mismatch.");
    f32 acc = 0.0f;
    size_t size = prod(lhs->lay->shape, lhs->lay->ndim);

#pragma omp parallel for reduction(+ : acc)
    for (size_t i = 0; i < size; i++) {
        f32 lval = lhs->ptr[offset_indexer(i, lhs->lay)];  // NOTE: stick to function call
        f32 rval = rhs->ptr[offset_indexer(i, rhs->lay)];
        acc += (lval * rval);
    }

    NDArray* out = array_empty((size_t[]){1}, 1);
    out->ptr[0] = acc;
    return out;
}

NDArray* array_reduce(
        binop acc_fn,
        const NDArray* src,
        const size_t* reduce_dims,
        size_t reduce_ndim,
        f32 acc_init) {
    char is_reduce_dim[src->lay->ndim];
    memset(is_reduce_dim, 0, src->lay->ndim);

    for (size_t i = 0; i < reduce_ndim; i++) {
        assert(reduce_dims[i] < src->lay->ndim && "ValueError: out of bounds");
        is_reduce_dim[reduce_dims[i]] = 1;
    }

    size_t dst_shape[src->lay->ndim];

    for (size_t i = 0; i < src->lay->ndim; i++)
        dst_shape[i] = is_reduce_dim[i] ? 1 : src->lay->shape[i];

    NDArray* dst = array_empty(dst_shape, src->lay->ndim);

    // NOTE: moving from src offset to dst offset simply zeros-out
    // any movement along reduced axes by setting stride=0.
    Layout* src_to_dst_layout = layout_copy(layout_alloc(src->lay->ndim), src->lay);

    size_t dst_size = prod(dst->lay->shape, dst->lay->ndim);
    size_t src_size = prod(src->lay->shape, src->lay->ndim);

    for (size_t i = 0; i < src->lay->ndim; i++)
        src_to_dst_layout->stride[i] = (is_reduce_dim[i]) ? 0 : dst->lay->stride[i];

    for (size_t i = 0; i < dst_size; i++)
        dst->ptr[offset_indexer(i, dst->lay)] = acc_init;

    f32* tmp_buffer = (f32*)alloc(sizeof(f32) * dst_size);
    for (size_t i = 0; i < dst_size; i++)
        tmp_buffer[i] = acc_init;

    for (size_t i = 0; i < dst_size; i++)
        dst->ptr[i] = acc_init;

#pragma omp parallel
    {
        f32* private_buffer = (f32*)malloc(sizeof(f32) * dst_size);
        for (size_t i = 0; i < dst_size; i++)
            private_buffer[i] = acc_init;

#pragma omp for
        for (size_t i = 0; i < src_size; i++) {
            size_t src_offset = offset_indexer(i, src->lay);
            f32 src_val = src->ptr[src_offset];

            size_t dst_offset = offset_indexer(i, src_to_dst_layout);
            private_buffer[dst_offset] = acc_fn(private_buffer[dst_offset], src_val);
        }

#pragma omp critical
        {
            for (size_t i = 0; i < dst_size; i++)
                dst->ptr[i] = acc_fn(dst->ptr[i], private_buffer[i]);
        }
        free(private_buffer);
    }

    return dst;
}

// -------------------------------------------------------------------------------------------------
// CONDITIONAL OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_array_array_where(const NDArray* cond, const NDArray* lhs, const NDArray* rhs) {
    assert(cond->lay->ndim == lhs->lay->ndim && lhs->lay->ndim && rhs->lay->ndim);
    for (size_t i = 0; i < lhs->lay->ndim; i++)
        assert(cond->lay->shape[i] == lhs->lay->shape[i]
               && lhs->lay->shape[i] == rhs->lay->shape[i]);

    NDArray* dst = array_empty(cond->lay->shape, cond->lay->ndim);
    size_t size = prod(cond->lay->shape, cond->lay->ndim);

#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        f32 lval = lhs->ptr[offset_indexer(i, lhs->lay)];
        f32 rval = rhs->ptr[offset_indexer(i, rhs->lay)];
        f32 cval = cond->ptr[offset_indexer(i, cond->lay)];
        dst->ptr[i] = cval ? lval : rval;
    }

    return dst;
}

NDArray* array_array_scalar_where(const NDArray* cond, const NDArray* lhs, f32 rhs) {
    assert(cond->lay->ndim == lhs->lay->ndim && lhs->lay->ndim);
    for (size_t i = 0; i < lhs->lay->ndim; i++)
        assert(cond->lay->shape[i] == lhs->lay->shape[i]);

    NDArray* dst = array_empty(lhs->lay->shape, lhs->lay->ndim);
    size_t size = prod(cond->lay->shape, cond->lay->ndim);

#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        f32 lval = lhs->ptr[offset_indexer(i, lhs->lay)];
        f32 cval = cond->ptr[offset_indexer(i, cond->lay)];
        dst->ptr[i] = cval ? lval : rhs;
    }

    return dst;
}

NDArray* array_scalar_array_where(const NDArray* cond, f32 lhs, const NDArray* rhs) {
    assert(cond->lay->ndim && rhs->lay->ndim);
    for (size_t i = 0; i < rhs->lay->ndim; i++)
        assert(cond->lay->shape[i] == rhs->lay->shape[i]);

    NDArray* dst = array_empty(rhs->lay->shape, rhs->lay->ndim);
    size_t size = prod(cond->lay->shape, cond->lay->ndim);

#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        f32 rval = rhs->ptr[offset_indexer(i, rhs->lay)];
        f32 cval = cond->ptr[offset_indexer(i, cond->lay)];
        dst->ptr[i] = cval ? lhs : rval;
    }
    return dst;
}

NDArray* array_scalar_scalar_where(const NDArray* cond, f32 lhs, f32 rhs) {
    NDArray* dst = array_empty(cond->lay->shape, cond->lay->ndim);
    size_t size = prod(cond->lay->shape, cond->lay->ndim);

#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
        dst->ptr[i] = cond->ptr[offset_indexer(i, cond->lay)] ? lhs : rhs;

    return dst;
}

// -------------------------------------------------------------------------------------------------
// JOIN OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_cat(const NDArray** arrays, size_t narray, const size_t* dims, size_t ndim) {
    size_t ref_ndim = arrays[0]->lay->ndim;
    for (size_t i = 1; i < narray; i++)
        assert(arrays[i]->lay->ndim == ref_ndim && "ValueError: dimension mismatch.");

    size_t* out_shape = size_t_copy(size_t_alloc(ref_ndim), arrays[0]->lay->shape, ref_ndim);

    for (size_t i = 0; i < ndim; i++) {
        size_t dim = dims[i];
        out_shape[dim] = 0;
        for (size_t j = 0; j < narray; j++)
            out_shape[dim] += arrays[j]->lay->shape[dim];
    }

    for (size_t i = 0; i < ref_ndim; i++) {
        bool is_concat_dim = false;
        for (size_t j = 0; j < ndim; j++) {
            if (dims[j] == i) {
                is_concat_dim = true;
                break;
            }
        }
        if (!is_concat_dim)
            for (size_t j = 1; j < narray; j++)
                assert(arrays[j]->lay->shape[i] == arrays[0]->lay->shape[i]
                       && "ValueError: non-concat dimensions must have the same size.");
    }

    NDArray* dst = array_zeros(out_shape, ref_ndim);
    FREE(out_shape);
    size_t* shape_offset = size_t_set(size_t_alloc(ndim), 0, ndim);

#pragma omp parallel for
    for (size_t i = 0; i < narray; i++) {
        size_t size = prod(arrays[i]->lay->shape, arrays[i]->lay->ndim);
        for (size_t j = 0; j < size; j++) {
            size_t* multi_index = index_unflatten(size_t_alloc(ref_ndim), j, arrays[i]->lay);
            for (size_t k = 0; k < ndim; k++)
                multi_index[dims[k]] += shape_offset[k];
            size_t src_offset = offset_indexer(j, arrays[i]->lay);
            size_t dst_offset = index_flatten(multi_index, dst->lay);
            dst->ptr[dst_offset] = arrays[i]->ptr[src_offset];
            FREE(multi_index);
        }

        for (size_t k = 0; k < ndim; k++)
            shape_offset[k] += arrays[i]->lay->shape[dims[k]];
    }

    FREE(shape_offset);
    return dst;
}