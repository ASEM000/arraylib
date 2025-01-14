
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

size_t prod(const size_t* nums, size_t ndim) {
    assert(ndim > 0 && "ValueError: non-positive ndim");
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++)
        total *= nums[i];
    return total;
}

size_t* size_t_alloc(size_t ndim) {
    assert(ndim > 0 && "ValueError: non-positive ndim");
    return (size_t*)alloc(sizeof(size_t) * ndim);
}

size_t* size_t_set(size_t* dst, size_t value, size_t size) {
    assert(value >= 0 && "ValueError: negative size");
    for (size_t i = 0; i < size; i++)
        dst[i] = value;
    return dst;
}

size_t* size_t_copy(size_t* dst, const size_t* src, size_t size) {
    assert(src != NULL && "ValueError: src copy is NULL");
    memcpy(dst, src, sizeof(size_t) * size);
    return dst;
}

size_t compute_flat_index(const size_t* index, const size_t* stride, size_t ndim) {
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

bool is_contiguous(const NDArray* array) {
    size_t contiguous_stride = 1;
    for (ssize_t i = array->lay->ndim - 1; i >= 0; i--) {
        if (contiguous_stride != array->lay->stride[i])
            return false;
        contiguous_stride *= array->lay->shape[i];
    }
    return true;
}

size_t cdiv(size_t a, size_t b) { return (a + b - 1) / b; }

size_t* compute_stride(size_t* dst, const size_t* shape, const size_t ndim) {
    dst[ndim - 1] = 1;
    for (ssize_t i = ndim - 2; i >= 0; i--)
        dst[i] = dst[i + 1] * shape[i + 1];
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

Layout* layout_copy(const Layout* src) {
    Layout* dst = (Layout*)alloc(sizeof(Layout));
    dst->ndim = src->ndim;
    dst->shape = size_t_copy(size_t_alloc(src->ndim), src->shape, src->ndim);
    dst->stride = size_t_copy(size_t_alloc(src->ndim), src->stride, src->ndim);
    return dst;
}

void layout_free(Layout* lay) {
    free(lay->shape);
    free(lay->stride);
    free(lay);
}

bool is_same_shape(const Layout* lhs, const Layout* rhs) {
    if (lhs->ndim != rhs->ndim)
        return false;
    for (size_t i = 0; i < lhs->ndim; i++)
        if (lhs->shape[i] != rhs->shape[i])
            return false;
    return false;
}

bool is_broadcastable(const Layout* lhs, const Layout* rhs) {
    ssize_t li = lhs->ndim;
    ssize_t ri = rhs->ndim;
    while (--li >= 0 && --ri >= 0)
        if ((lhs->shape[li] != rhs->shape[ri]) && lhs->shape[li] != 1 && rhs->shape[ri] != 1)
            return false;
    return true;
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
// ITERATOR
// -------------------------------------------------------------------------------------------------

NDIter* iter_create(f32* ptr, const Layout* lay) {
    NDIter* iter = (NDIter*)alloc(sizeof(NDIter));
    iter->lay = layout_copy(lay);
    iter->index = size_t_set(size_t_alloc(lay->ndim), 0, lay->ndim);
    iter->ptr = ptr;
    iter->status = ITER_START;
    return iter;
}

void iter_free(NDIter* iter) {
    layout_free(iter->lay);
    free(iter->index);
    free(iter);
}

bool iter_next(NDIter* iter) {
    if (iter->status == ITER_END)
        return false;

    if (iter->status == ITER_START) {
        iter->status = ITER_RUN;
        return true;
    }

    for (ssize_t i = iter->lay->ndim - 1; i >= 0; i--) {
        iter->index[i]++;
        if (iter->index[i] < iter->lay->shape[i]) {
            // NOTE: to skip a dimension set the shape and stride to 0
            // NOTE: to broadcast a dimension set the shape to the broadcasted value
            // and the stride to 0
            iter->ptr += iter->lay->stride[i];
            return true;
        }
        iter->index[i] = 0;  // move to next
        iter->ptr -= (iter->lay->shape[i] - 1) * iter->lay->stride[i];
    }
    iter->status = ITER_END;
    return false;
}

// -------------------------------------------------------------------------------------------------
// COPY
// -------------------------------------------------------------------------------------------------

NDArray* array_shallow_copy(NDArray* src) {
    assert(src != NULL && "TypeError: shallow copy of NULL.");
    NDArray* dst = (NDArray*)alloc(sizeof(NDArray));
    dst->ptr = src->ptr;
    dst->lay = layout_copy(src->lay);
    dst->data = src->data;
    src->data->refs++;
    dst->view = true;
    return dst;
}

NDArray* array_deep_copy(NDArray* src) {
    NDArray* dst = array_empty(src->lay->shape, src->lay->ndim);
    NDIter* idst = iter_create(dst->ptr, dst->lay);
    NDIter* isrc = iter_create(src->ptr, src->lay);
    while (iter_next(idst) && iter_next(isrc))
        *idst->ptr = *isrc->ptr;
    FREE(idst);
    FREE(isrc);
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

f32 array_get_scalar_from_index(NDArray* array, const size_t* index) {
    size_t findex = compute_flat_index(index, array->lay->stride, array->lay->ndim);
    return array->ptr[findex];
}

NDArray* array_get_view_from_range(
        NDArray* src,
        const size_t* start,
        const size_t* end,
        const size_t* step) {
    for (size_t i = 0; i < src->lay->ndim; i++) {
        assert(start[i] < end[i] && start[i] >= 0 && "ValueError: invalid start.");
        assert(end[i] <= src->lay->shape[i] && "ValueError: invald end.");
        assert(step[i] > 0 && "ValueError: negative step.");
    }
    Layout* view_layout = layout_alloc(src->lay->ndim);
    size_t offset = 0;
    for (size_t i = 0; i < src->lay->ndim; i++) {
        view_layout->shape[i] = cdiv(end[i] - start[i], step[i]);
        view_layout->stride[i] = src->lay->stride[i] * step[i];
        offset += start[i] * src->lay->stride[i];
    }
    assert(offset < src->data->size && "ValueError: offset >= size.");
    NDArray* dst = array_shallow_copy(src);
    FREE(dst->lay);
    dst->lay = view_layout;
    dst->ptr = src->ptr + offset;
    return dst;
}

// -------------------------------------------------------------------------------------------------
// SETTERS
// -------------------------------------------------------------------------------------------------

NDArray* array_set_scalar_from_index(NDArray* array, f32 value, const size_t* index) {
    size_t findex = compute_flat_index(index, array->lay->stride, array->lay->ndim);
    array->ptr[findex] = value;
    return array;
}

NDArray* array_set_scalar_from_range(
        NDArray* array,
        f32 value,
        const size_t* start,
        const size_t* end,
        const size_t* step) {
    for (size_t i = 0; i < array->lay->ndim; i++) {
        assert(start[i] < end[i] && "ValueError: start index > end index.");
        assert(start[i] >= 0 && "ValueError: negative start index");
        assert(end[i] <= array->lay->shape[i] && "ValueError: end index out of bounds.");
        assert(step[i] > 0 && "ValueError: negative step size.");
    }
    size_t ndim = array->lay->ndim;
    Layout* view_layout = layout_alloc(array->lay->ndim);
    for (size_t i = 0; i < ndim; i++) {
        view_layout->shape[i] = cdiv(end[i] - start[i], step[i]);
        view_layout->shape[i] = array->lay->stride[i] * step[i];
    }

    f32* ptr = array->ptr + compute_flat_index(start, array->lay->stride, array->lay->ndim);
    NDIter* iter = iter_create(ptr, view_layout);
    while (iter_next(iter))
        *iter->ptr = value;
    FREE(iter);
    return array;
}

NDArray* array_set_view_from_array(
        NDArray* dst,
        NDArray* src,
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
    Layout* view_layout = layout_alloc(ndim);
    view_layout->ndim = ndim;
    for (size_t i = 0; i < ndim; i++) {
        view_layout->shape[i] = cdiv(end[i] - start[i], step[i]);
        view_layout->stride[i] = dst->lay->stride[i] * step[i];
    }

    assert((prod(view_layout->shape, ndim) == prod(src->lay->shape, ndim))
           && "ValueError: set value shape mismatch.");
    f32* pdst = dst->ptr + compute_flat_index(start, dst->lay->stride, ndim);
    NDIter* idst = iter_create(pdst, view_layout);
    NDIter* isrc = iter_create(src->ptr, src->lay);
    while (iter_next(isrc) && iter_next(idst))
        *idst->ptr = *isrc->ptr;
    FREE(isrc);
    FREE(idst);
    FREE(view_layout);
    return dst;
}

// -------------------------------------------------------------------------------------------------
// RESHAPING
// -------------------------------------------------------------------------------------------------

NDArray* array_reshape(NDArray* src, const size_t* shape, size_t ndim) {
    assert(prod(shape, ndim) == src->data->size);
    NDArray* dst = is_contiguous(src) ? array_shallow_copy(src) : array_deep_copy(src);
    FREE(dst->lay);
    dst->lay = layout_alloc(ndim);
    dst->lay->shape = size_t_copy(dst->lay->shape, shape, ndim);
    dst->lay->stride = compute_stride(dst->lay->stride, shape, ndim);
    dst->lay->ndim = ndim;
    return dst;
}

NDArray* array_transpose(NDArray* src, const size_t* dims) {
    NDArray* dst = is_contiguous(src) ? array_shallow_copy(src) : array_deep_copy(src);
    size_t ndim = src->lay->ndim;
    Layout* original_layout = layout_copy(src->lay);
    for (size_t i = 0; i < ndim; i++) {
        dst->lay->shape[i] = original_layout->shape[dims[i]];
        dst->lay->stride[i] = original_layout->stride[dims[i]];
    }
    FREE(original_layout);
    return dst;
}

NDArray* array_move_axis(NDArray* src, const size_t* from, const size_t* to, size_t ndim) {
    for (size_t i = 0; i < ndim; i++) {
        assert(from[i] >= 0 && from[i] < src->lay->ndim && "ValueError: out of bounds");
        assert(from[i] >= 0 && from[i] < src->lay->ndim && "ValueError: out of bounds");
    }

    char bucket[src->lay->ndim];
    memset(bucket, 0, src->lay->ndim);

    for (size_t i = 0; i < ndim; i++)
        bucket[from[i]] = 1;  // mark used axis
    NDArray* dst = is_contiguous(src) ? array_shallow_copy(src) : array_deep_copy(src);
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

NDArray* array_ravel(NDArray* src) {
    size_t total = prod(src->lay->shape, src->lay->ndim);
    NDArray* dst = is_contiguous(src) ? array_shallow_copy(src) : array_deep_copy(src);
    FREE(dst->lay);
    dst->lay = layout_alloc(1);
    dst->lay->shape = size_t_set(size_t_alloc(1), total, 1);
    dst->lay->stride = size_t_set(size_t_alloc(1), 1, 1);
    return dst;
}

// -------------------------------------------------------------------------------------------------
// ARRAY-SCALAR OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_scalar_op(binop fn, NDArray* src, f32 rhs) {
    NDArray* dst = array_empty(src->lay->shape, src->lay->ndim);
    NDIter* isrc = iter_create(src->ptr, src->lay);
    NDIter* idst = iter_create(dst->ptr, dst->lay);
    while (iter_next(isrc) && iter_next(idst))
        *idst->ptr = fn(*isrc->ptr, rhs);
    FREE(isrc);
    FREE(idst);
    return dst;
}

NDArray* array_scalar_add(NDArray* lhs, f32 rhs) { return array_scalar_op(sum32, lhs, rhs); }
NDArray* array_scalar_sub(NDArray* lhs, f32 rhs) { return array_scalar_op(sub32, lhs, rhs); }
NDArray* array_scalar_mul(NDArray* lhs, f32 rhs) { return array_scalar_op(mul32, lhs, rhs); }
NDArray* array_scalar_div(NDArray* lhs, f32 rhs) { return array_scalar_op(div32, lhs, rhs); }
NDArray* array_scalar_pow(NDArray* lhs, f32 rhs) { return array_scalar_op(pow32, lhs, rhs); }

NDArray* array_scalar_eq(NDArray* lhs, f32 rhs) { return array_scalar_op(eq32, lhs, rhs); }
NDArray* array_scalar_neq(NDArray* lhs, f32 rhs) { return array_scalar_op(neq32, lhs, rhs); }
NDArray* array_scalar_lt(NDArray* lhs, f32 rhs) { return array_scalar_op(lt32, lhs, rhs); }
NDArray* array_scalar_leq(NDArray* lhs, f32 rhs) { return array_scalar_op(leq32, lhs, rhs); }
NDArray* array_scalar_gt(NDArray* lhs, f32 rhs) { return array_scalar_op(gt32, lhs, rhs); }
NDArray* array_scalar_geq(NDArray* lhs, f32 rhs) { return array_scalar_op(geq32, lhs, rhs); }

// -------------------------------------------------------------------------------------------------
// MATMUL
// -------------------------------------------------------------------------------------------------

NDArray* array_array_matmul(NDArray* lhs, NDArray* rhs) {
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
                            f32 res_val = mul32(lhs_val, rhs_val);
                            f32 out_val = pout[i * out_s0 + j * out_s1];
                            pout[i * out_s0 + j * out_s1] = sum32(out_val, res_val);
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

NDArray* array_array_scalar_op(binop fn, NDArray* lhs, NDArray* rhs) {
    assert(is_broadcastable(lhs->lay, rhs->lay) && "ValueError: can not broadcast.");
    size_t max_ndim = lhs->lay->ndim > rhs->lay->ndim ? lhs->lay->ndim : rhs->lay->ndim;
    Layout* lhs_blay = layout_alloc(max_ndim);
    Layout* rhs_blay = layout_alloc(max_ndim);

    for (size_t i = 0; i < max_ndim; i++) {
        size_t oi = (max_ndim - i - 1);
        size_t li = (i < lhs->lay->ndim) ? lhs->lay->ndim - i - 1 : SIZE_MAX;
        size_t ri = (i < rhs->lay->ndim) ? rhs->lay->ndim - i - 1 : SIZE_MAX;
        size_t lsi = (li == SIZE_MAX) ? 1 : lhs->lay->shape[li];
        size_t rsi = (ri == SIZE_MAX) ? 1 : rhs->lay->shape[ri];
        size_t max_shape = lsi > rsi ? lsi : rsi;
        lhs_blay->shape[oi] = rhs_blay->shape[oi] = max_shape;
        lhs_blay->stride[oi] = (li >= 0 && lsi == max_shape) ? lhs->lay->stride[li] : 0;
        rhs_blay->stride[oi] = (ri >= 0 && rsi == max_shape) ? rhs->lay->stride[ri] : 0;
    }
    NDArray* out = array_empty(lhs_blay->shape, lhs_blay->ndim);
    NDIter* ilhs = iter_create(lhs->ptr, lhs_blay);
    NDIter* irhs = iter_create(rhs->ptr, rhs_blay);
    NDIter* iout = iter_create(out->ptr, out->lay);

    while (iter_next(ilhs) && iter_next(irhs) && iter_next(iout))
        *iout->ptr = fn(*ilhs->ptr, *irhs->ptr);

    FREE(ilhs);
    FREE(irhs);
    FREE(iout);
    FREE(lhs_blay);
    FREE(rhs_blay);
    return out;
}

NDArray* array_array_sum(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(sum32, lhs, rhs);
}
NDArray* array_array_sub(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(sub32, lhs, rhs);
}
NDArray* array_array_mul(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(mul32, lhs, rhs);
}
NDArray* array_array_div(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(div32, lhs, rhs);
}
NDArray* array_array_pow(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(pow32, lhs, rhs);
}

// comparison
NDArray* array_array_eq(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(eq32, lhs, rhs);
}
NDArray* array_array_neq(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(neq32, lhs, rhs);
}
NDArray* array_array_gt(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(gt32, lhs, rhs);
}
NDArray* array_array_geq(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(geq32, lhs, rhs);
}
NDArray* array_array_lt(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(lt32, lhs, rhs);
}
NDArray* array_array_leq(NDArray* lhs, NDArray* rhs) {
    return array_array_scalar_op(leq32, lhs, rhs);
}

// -------------------------------------------------------------------------------------------------
// ELEMENTWISE OPERATIONS
// -------------------------------------------------------------------------------------------------

static inline f32 log32(f32 lhs) { return lhs == 0 ? 0 : logf(lhs); }
static inline f32 neg32(f32 lhs) { return -lhs; }
static inline f32 exp32(f32 lhs) { return expf(lhs); }

NDArray* array_op(uniop fn, NDArray* src) {
    NDArray* dst = array_empty(src->lay->shape, src->lay->ndim);
    NDIter* isrc = iter_create(src->ptr, src->lay);
    NDIter* idst = iter_create(dst->ptr, dst->lay);
    while (iter_next(isrc) && iter_next(idst))
        *idst->ptr = fn(*isrc->ptr);
    FREE(isrc);
    FREE(idst);
    return dst;
}

NDArray* array_log(NDArray* array) { return array_op(log32, array); }
NDArray* array_neg(NDArray* array) { return array_op(neg32, array); }
NDArray* array_exp(NDArray* array) { return array_op(exp32, array); }

// -------------------------------------------------------------------------------------------------
// REDUCTION OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_array_dot(NDArray* lhs, NDArray* rhs) {
    assert(lhs->lay->ndim == 1 && "ValueError: lhs dimension != 1 for dot.");
    assert(rhs->lay->ndim == 1 && "ValueError: rhs dimension != 1 for dot.");
    assert(lhs->lay->shape[0] == rhs->lay->shape[0] && "ValueError: dot shape mismatch.");
    f32 acc = 0.0f;
    NDIter* ilhs = iter_create(lhs->ptr, lhs->lay);
    NDIter* irhs = iter_create(rhs->ptr, rhs->lay);
    while (iter_next(ilhs) && iter_next(irhs))
        acc = sum32(acc, mul32(*ilhs->ptr, *irhs->ptr));
    FREE(ilhs);
    FREE(irhs);
    NDArray* out = array_empty((size_t[]){1}, 1);
    out->data->mem[0] = acc;
    return out;
}

NDArray* array_reduce(
        binop acc_fn,
        NDArray* src,
        const size_t* reduce_dims,
        size_t reduce_ndim,
        f32 acc_init) {
    bool is_reduce_dim[src->lay->ndim];
    memset(is_reduce_dim, 0, src->lay->ndim);
    for (size_t i = 0; i < reduce_ndim; i++)
        is_reduce_dim[reduce_dims[i]] = 1;

    size_t dst_shape[src->lay->ndim];
    for (size_t i = 0; i < src->lay->ndim; i++)
        dst_shape[i] = (is_reduce_dim[i] == true) ? 1 : src->lay->shape[i];

    Layout* reduced_lay = layout_copy(src->lay);
    for (size_t i = 0; i < src->lay->ndim; i++) {
        reduced_lay->stride[i] = (is_reduce_dim[i] == true) ? src->lay->stride[i] : 0;
        reduced_lay->shape[i] = (is_reduce_dim[i] == true) ? src->lay->shape[i] : 0;
    }

    NDArray* dst = array_zeros(dst_shape, src->lay->ndim);
    NDIter* idst = iter_create(dst->ptr, dst->lay);
    NDIter* isrc = iter_create(src->ptr, reduced_lay);

    while (iter_next(idst)) {
        f32 sum = acc_init;
        size_t offset = compute_flat_index(idst->index, src->lay->stride, src->lay->ndim);
        isrc->ptr = src->ptr + offset;
        isrc->index = size_t_copy(isrc->index, idst->index, src->lay->ndim);
        while (iter_next(isrc))
            sum = acc_fn(sum, *isrc->ptr);
        isrc->status = ITER_START;
        array_set_scalar_from_index(dst, sum, idst->index);
    }
    FREE(isrc);
    FREE(idst);
    return dst;
}

NDArray* array_reduce_max(NDArray* array, size_t* reduce_dims, size_t ndim) {
    return array_reduce(max32, array, reduce_dims, ndim, -INFINITY);
}

NDArray* array_reduce_min(NDArray* array, size_t* reduce_dims, size_t ndim) {
    return array_reduce(min32, array, reduce_dims, ndim, INFINITY);
}

NDArray* array_reduce_sum(NDArray* array, size_t* reduce_dims, size_t ndim) {
    return array_reduce(sum32, array, reduce_dims, ndim, 0);
}

// -------------------------------------------------------------------------------------------------
// CONDITIONAL OPERATIONS
// -------------------------------------------------------------------------------------------------

NDArray* array_array_array_where(NDArray* cond, NDArray* lhs, NDArray* rhs) {
    assert(cond->lay->ndim == lhs->lay->ndim && lhs->lay->ndim && rhs->lay->ndim);
    for (size_t i = 0; i < lhs->lay->ndim; i++)
        assert(cond->lay->shape[i] == lhs->lay->shape[i]
               && lhs->lay->shape[i] == rhs->lay->shape[i]);

    NDArray* dst = array_empty(cond->lay->shape, cond->lay->ndim);
    NDIter* ilhs = iter_create(lhs->ptr, lhs->lay);
    NDIter* irhs = iter_create(rhs->ptr, rhs->lay);
    NDIter* icond = iter_create(cond->ptr, cond->lay);
    NDIter* idst = iter_create(dst->ptr, dst->lay);

    while (iter_next(ilhs) && iter_next(irhs) && iter_next(icond) && iter_next(idst))
        *idst->ptr = *icond->ptr ? *ilhs->ptr : *irhs->ptr;

    FREE(ilhs);
    FREE(irhs);
    FREE(icond);
    FREE(idst);
    return dst;
}

NDArray* array_array_scalar_where(NDArray* cond, NDArray* lhs, f32 rhs) {
    assert(cond->lay->ndim == lhs->lay->ndim && lhs->lay->ndim);
    for (size_t i = 0; i < lhs->lay->ndim; i++)
        assert(cond->lay->shape[i] == lhs->lay->shape[i]);

    NDArray* dst = array_empty(lhs->lay->shape, lhs->lay->ndim);
    NDIter* ilhs = iter_create(lhs->ptr, lhs->lay);
    NDIter* icond = iter_create(cond->ptr, cond->lay);
    NDIter* idst = iter_create(dst->ptr, dst->lay);

    while (iter_next(ilhs) && iter_next(icond) && iter_next(idst))
        *idst->ptr = *icond->ptr ? *ilhs->ptr : rhs;

    FREE(ilhs);
    FREE(icond);
    FREE(idst);
    return dst;
}

NDArray* array_scalar_array_where(NDArray* cond, f32 lhs, NDArray* rhs) {
    assert(cond->lay->ndim && rhs->lay->ndim);
    for (size_t i = 0; i < rhs->lay->ndim; i++)
        assert(cond->lay->shape[i] == rhs->lay->shape[i]);

    NDArray* dst = array_empty(rhs->lay->shape, rhs->lay->ndim);
    NDIter* irhs = iter_create(rhs->ptr, rhs->lay);
    NDIter* icond = iter_create(cond->ptr, cond->lay);
    NDIter* idst = iter_create(dst->ptr, dst->lay);

    while (iter_next(irhs) && iter_next(icond) && iter_next(idst))
        *idst->ptr = *icond->ptr ? lhs : *irhs->ptr;

    FREE(irhs);
    FREE(icond);
    FREE(idst);
    return dst;
}

NDArray* array_scalar_scalar_where(NDArray* cond, f32 lhs, f32 rhs) {
    NDArray* dst = array_empty(cond->lay->shape, cond->lay->ndim);
    NDIter* icond = iter_create(cond->ptr, cond->lay);
    NDIter* idst = iter_create(dst->ptr, dst->lay);

    while (iter_next(icond) && iter_next(idst))
        *idst->ptr = *icond->ptr ? lhs : rhs;

    FREE(icond);
    FREE(idst);
    return dst;
}
