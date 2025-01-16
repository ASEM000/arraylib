#include <setjmp.h>
#include <signal.h>
#include "arraylib.h"

jmp_buf jump_buffer;
int assertion_failures = 0;

void handle_sigabrt(int signum) {
    longjmp(jump_buffer, 1);  // Jump back to the setjmp point
}

#define TEST(expr)                   \
    signal(SIGABRT, handle_sigabrt); \
    if (setjmp(jump_buffer) == 0) {  \
        expr;                        \
    } else {                         \
        assertion_failures++;        \
    }

void test_alloc() {
    void* ptr = alloc(10);
    TEST(assert(ptr != NULL));
}

void test_utils() {
    size_t nums1[] = {1, 2, 3};
    TEST(assert(prod(nums1, 3) == 6));

    //  +1
    size_t nums2[] = {1, 2, 3};
    TEST(prod(nums2, 0));

    size_t* arr = size_t_alloc(5);
    TEST(assert(arr != NULL));
    free(arr);

    size_t arr2[5];
    TEST(assert(size_t_set(arr2, 42, 5) == arr2));
    TEST(assert(arr2[0] == 42 && arr2[4] == 42));

    size_t src[] = {1, 2, 3};
    size_t dst[3];
    TEST(assert(size_t_copy(dst, src, 3) == dst));
    TEST(assert(dst[0] == 1 && dst[2] == 3));

    size_t index[] = {1, 1};
    size_t stride[] = {3, 1};
    TEST(assert(compute_flat_index(index, stride, 2) == 4));

    TEST(assert(clamp(5, 0, 10) == 5));
    TEST(assert(clamp(-1, 0, 10) == 0));
    TEST(assert(clamp(11, 0, 10) == 10));
}

void test_array_creation() {
    size_t shape[] = {2, 3};
    NDArray* array = array_empty(shape, 2);
    TEST(assert(array != NULL));
    TEST(assert(array->lay->ndim == 2));
    TEST(assert(array->lay->shape[0] == 2 && array->lay->shape[1] == 3));
    FREE(array);
}

void test_iterator() {
    size_t shape[] = {2, 3};
    NDArray* array = array_empty(shape, 2);
    NDIter* iter = iter_create(array->ptr, array->lay);
    TEST(assert(iter->ptr != NULL));

    size_t count = 0;
    while (iter_next(iter))
        count++;
    TEST(assert(count == 6));
    FREE(array);
    FREE(iter);
}

void test_array_initialization() {
    size_t shape[] = {2, 2};
    NDArray* zeros = array_zeros(shape, 2);
    TEST(assert(zeros != NULL));
    TEST(assert(zeros->data->mem[0] == 0 && zeros->data->mem[3] == 0));
    FREE(zeros);

    f32 elems[] = {1, 2, 3, 4};
    NDArray* filled = array_fill(elems, shape, 2);
    TEST(assert(filled != NULL));
    TEST(assert(filled->data->mem[0] == 1 && filled->data->mem[3] == 4));
    FREE(filled);

    NDArray* ones = array_ones(shape, 2);
    TEST(assert(ones != NULL));
    TEST(assert(ones->data->mem[0] == 1 && ones->data->mem[3] == 1));
    FREE(ones);

    NDArray* arange = array_arange(0, 5, 1);
    TEST(assert(arange != NULL));
    TEST(assert(arange->data->mem[0] == 0 && arange->data->mem[4] == 4));
    FREE(arange);

    NDArray* linspace = array_linspace(0, 1, 5);
    TEST(assert(linspace != NULL));
    TEST(assert(linspace->data->mem[0] == 0 && linspace->data->mem[4] == 1));
    FREE(linspace);
}

void test_array_operations() {
    size_t shape[] = {2, 2};
    NDArray* array = array_ones(shape, 2);
    NDArray* result = array_scalar_add(array, 1);
    TEST(assert(result != NULL));
    TEST(assert(result->data->mem[0] == 2 && result->data->mem[3] == 2));
    FREE(array);
    FREE(result);

    NDArray* array1 = array_ones(shape, 2);
    NDArray* array2 = array_ones(shape, 2);
    NDArray* sum = array_array_sum(array1, array2);
    TEST(assert(sum != NULL));
    TEST(assert(sum->data->mem[0] == 2 && sum->data->mem[3] == 2));
    FREE(array1);
    FREE(array2);
    FREE(sum);
}

void test_reduction_operations() {
    size_t shape[] = {3};
    NDArray* array1 = array_fill((f32[]){1, 2, 3}, shape, 1);
    NDArray* array2 = array_fill((f32[]){4, 5, 6}, shape, 1);
    NDArray* dot = array_array_dot(array1, array2);
    TEST(assert(dot != NULL));
    TEST(assert(dot->data->mem[0] == 32));
    FREE(array1);
    FREE(array2);
    FREE(dot);
}

void test_view_from_range() {
    size_t shape[] = {3, 3};
    NDArray* array = array_empty(shape, 2);

    for (size_t i = 0; i < 9; i++) {
        array->data->mem[i] = (f32)i;
    }

    size_t start[] = {1, 1};
    size_t end[] = {3, 3};
    size_t step[] = {1, 1};
    NDArray* view = array_get_view_from_range(array, start, end, step);

    TEST(assert(view->lay->shape[0] == 2 && view->lay->shape[1] == 2));
    TEST(assert(view->ptr[0] == 4.0f));  // [1, 1]
    TEST(assert(view->ptr[1] == 5.0f));  // [1, 2]
    TEST(assert(view->ptr[3] == 7.0f));  // [2, 1]
    TEST(assert(view->ptr[4] == 8.0f));  // [2, 2]

    // Free the view and original array
    FREE(view);
    FREE(array);
}

void test_modify_view() {
    size_t shape[] = {3, 3};
    NDArray* array = array_empty(shape, 2);

    memset(array->data->mem, 0, sizeof(f32) * 9);

    size_t start[] = {0, 0};
    size_t end[] = {1, 3};
    size_t step[] = {1, 1};
    NDArray* view = array_get_view_from_range(array, start, end, step);

    array_set_scalar_from_index(view, 42.0f, (size_t[]){0, 1});

    TEST(assert(array->data->mem[1] == 42.0f));

    FREE(view);
    FREE(array);
}

void test_non_contiguous_view() {
    size_t shape[] = {4};
    NDArray* array = array_empty(shape, 1);

    for (size_t i = 0; i < 4; i++) {
        array->data->mem[i] = (f32)i;
    }

    size_t start[] = {0};
    size_t end[] = {4};
    size_t step[] = {2};
    NDArray* view = array_get_view_from_range(array, start, end, step);

    TEST(assert(view->lay->shape[0] == 2));
    TEST(assert(view->ptr[0] == 0.0f));  // [0]
    TEST(assert(view->ptr[2] == 2.0f));  // [2]

    FREE(view);
    FREE(array);
}

void test_basic_matmul() {
    size_t shape[] = {2, 2};
    NDArray* lhs = array_empty(shape, 2);
    NDArray* rhs = array_empty(shape, 2);

    lhs->data->mem[0] = 1.0f;
    lhs->data->mem[1] = 2.0f;
    lhs->data->mem[2] = 3.0f;
    lhs->data->mem[3] = 4.0f;

    rhs->data->mem[0] = 5.0f;
    rhs->data->mem[1] = 6.0f;
    rhs->data->mem[2] = 7.0f;
    rhs->data->mem[3] = 8.0f;
    NDArray* result = array_array_matmul(lhs, rhs);

    TEST(assert(result != NULL));
    TEST(assert(result->lay->shape[0] == 2 && result->lay->shape[1] == 2));
    TEST(assert(result->data->mem[0] == 19.0f));  // [1*5 + 2*7]
    TEST(assert(result->data->mem[1] == 22.0f));  // [1*6 + 2*8]
    TEST(assert(result->data->mem[2] == 43.0f));  // [3*5 + 4*7]
    TEST(assert(result->data->mem[3] == 50.0f));  // [3*6 + 4*8]

    FREE(lhs);
    FREE(rhs);
    FREE(result);
}

void test_dimension_mismatch() {
    size_t shape1[] = {2, 3};
    size_t shape2[] = {4, 2};
    NDArray* lhs = array_empty(shape1, 2);
    NDArray* rhs = array_empty(shape2, 2);

    //  + 1
    TEST(array_array_matmul(lhs, rhs));

    FREE(lhs);
    FREE(rhs);
}

void test_matmul_identity() {
    size_t shape[] = {2, 2};
    NDArray* matrix = array_empty(shape, 2);
    matrix->data->mem[0] = 1.0f;
    matrix->data->mem[1] = 2.0f;
    matrix->data->mem[2] = 3.0f;
    matrix->data->mem[3] = 4.0f;

    NDArray* identity = array_empty(shape, 2);
    identity->data->mem[0] = 1.0f;
    identity->data->mem[1] = 0.0f;
    identity->data->mem[2] = 0.0f;
    identity->data->mem[3] = 1.0f;

    NDArray* result = array_array_matmul(matrix, identity);

    TEST(assert(result != NULL));
    TEST(assert(result->lay->shape[0] == 2 && result->lay->shape[1] == 2));
    TEST(assert(result->data->mem[0] == 1.0f));
    TEST(assert(result->data->mem[1] == 2.0f));
    TEST(assert(result->data->mem[2] == 3.0f));
    TEST(assert(result->data->mem[3] == 4.0f));

    // Free the matrices
    FREE(matrix);
    FREE(identity);
    FREE(result);
}

void test_blocked_matmul() {
    size_t shape[] = {4, 4};
    NDArray* lhs = array_empty(shape, 2);
    NDArray* rhs = array_empty(shape, 2);

    for (size_t i = 0; i < 16; i++) {
        lhs->data->mem[i] = (f32)i;
    }

    for (size_t i = 0; i < 16; i++) {
        rhs->data->mem[i] = (f32)i;
    }

    NDArray* result = array_array_matmul(lhs, rhs);

    TEST(assert(result != NULL));
    TEST(assert(result->lay->shape[0] == 4 && result->lay->shape[1] == 4));

    f32 expected[] = {
            56.0f,
            62.0f,
            68.0f,
            74.0f,
            152.0f,
            174.0f,
            196.0f,
            218.0f,
            248.0f,
            286.0f,
            324.0f,
            362.0f,
            344.0f,
            398.0f,
            452.0f,
            506.0f};

    for (size_t i = 0; i < 16; i++) {
        TEST(assert(result->data->mem[i] == expected[i]));
    }

    FREE(lhs);
    FREE(rhs);
    FREE(result);
}

void test_array_reduce_max() {
    NDArray* array = array_fill((f32[]){1, 5, 3, 2}, (size_t[]){4}, 1);
    NDArray* result = array_reduce_max(array, (size_t[]){0}, 1);
    TEST(assert(result->data->mem[0] == 5 && "Test case 1 failed"));
    array = array_fill((f32[]){-1, -5, -3, -2}, (size_t[]){4}, 1);
    result = array_reduce_max(array, (size_t[]){0}, 1);
    TEST(assert(result->data->mem[0] == -1 && "Test case 2 failed"));
    FREE(array);
    FREE(result);
}

void test_array_reduce_sum() {
    NDArray* array = array_arange(1, 26, 1);
    NDArray* reshaped_array = array_reshape(array, (size_t[]){5, 5}, 2);
    NDArray* reduce_array = array_reduce_sum(reshaped_array, (size_t[]){0}, 1);

    TEST(assert(reduce_array->data->mem[0] == 55.));
    TEST(assert(reduce_array->lay->ndim == 2));
    TEST(assert(reduce_array->lay->shape[0] == 1));
    TEST(assert(reduce_array->lay->shape[1] == 5));
    FREE(array);
    FREE(reshaped_array);
    FREE(reduce_array);

    NDArray* array_ = array_arange(1, 3 * 4 * 5 * 6 + 1, 1);
    NDArray* reshaped_array_ = array_reshape(array_, (size_t[]){3, 4, 5, 6}, 4);
    NDArray* reduced_array_ = array_reduce_sum(reshaped_array_, (size_t[]){2}, 1);
    TEST(assert(reduced_array_->data->mem[3] == 80.));
    FREE(array_);
    FREE(reshaped_array_);
    FREE(reduced_array_);
}

void test_array_reshape() {
    NDArray* array = array_arange(1, 3 * 4 * 5 * 6 + 1, 1);
    NDArray* reshaped_array = array_reshape(array, (size_t[]){3, 4, 5, 6}, 4);
    size_t src[2] = {1, 3};
    size_t dst[2] = {0, 2};
    array_move_dim(reshaped_array, src, dst, 2);
    FREE(array);
    FREE(reshaped_array);
}

void test_move_dim() {
    size_t shape[] = {2, 3};
    NDArray* array = array_zeros(shape, 2);
    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 3; j++) {
            size_t index[] = {i, j};
            array_set_scalar_from_index(array, i * 3 + j, index);
        }
    }

    size_t src[] = {0, 1};
    size_t dst[] = {1, 0};
    NDArray* moved_array = array_move_dim(array, src, dst, 2);

    assert(moved_array->lay->shape[0] == 3 && moved_array->lay->shape[1] == 2);
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 2; j++) {
            size_t index[] = {i, j};
            f32 value = array_get_scalar_from_index(moved_array, index);
            assert(value == j * 3 + i);
        }
    }
    FREE(array);
    FREE(moved_array);
}

void test_transpose() {
    size_t shape[] = {2, 3, 4};
    NDArray* array = array_zeros(shape, 3);
    array = array_transpose(array, (size_t[]){1, 0, 2});
    FREE(array);
}

void test_stack() {
    NDArray* lhs = array_zeros((size_t[]){2, 3}, 2);
    NDArray* rhs = array_ones((size_t[]){2, 5}, 2);
    const NDArray* arrays[] = {lhs, rhs};
    NDArray* out = array_cat(arrays, 2, (size_t[]){1}, 1);
    FREE(lhs);
    FREE(rhs);
    FREE(out);
}

void test_layout_broadcast_multiple() {
    // 3x2x4 => 3x2x4
    //     4 => 3x2x4
    //   2x4 => 3x2x4
    // 3x1x1 => 3x2x4

    Layout* lay1 = layout_alloc(3);
    lay1->shape[0] = 3, lay1->shape[1] = 2, lay1->shape[2] = 4;
    lay1->stride[0] = 8, lay1->stride[1] = 4, lay1->stride[2] = 1;

    Layout* lay2 = layout_alloc(1);
    lay2->shape[0] = 4;
    lay2->stride[0] = 1;

    Layout* lay3 = layout_alloc(2);
    lay3->shape[0] = 2, lay3->shape[1] = 4;
    lay3->stride[0] = 4, lay3->stride[1] = 1;

    Layout* lay4 = layout_alloc(3);
    lay4->shape[0] = 3, lay4->shape[1] = 1, lay4->shape[2] = 1;
    lay4->stride[0] = 1, lay4->stride[1] = 1, lay4->stride[2] = 1;

    const Layout* layouts[] = {lay1, lay2, lay3, lay4};
    Layout** blays = layout_broadcast(layouts, 4);

    assert(blays[0]->ndim == 3);
    assert(blays[0]->shape[0] == 3);
    assert(blays[0]->shape[1] == 2);
    assert(blays[0]->shape[2] == 4);
    assert(blays[0]->stride[0] == 8);
    assert(blays[0]->stride[1] == 4);
    assert(blays[0]->stride[2] == 1);

    assert(blays[1]->ndim == 3);
    assert(blays[1]->shape[0] == 3);
    assert(blays[1]->shape[1] == 2);
    assert(blays[1]->shape[2] == 4);
    assert(blays[1]->stride[0] == 0);
    assert(blays[1]->stride[1] == 0);
    assert(blays[1]->stride[2] == 1);

    assert(blays[2]->ndim == 3);
    assert(blays[2]->shape[0] == 3);
    assert(blays[2]->shape[1] == 2);
    assert(blays[2]->shape[2] == 4);
    assert(blays[2]->stride[0] == 0);
    assert(blays[2]->stride[1] == 4);
    assert(blays[2]->stride[2] == 1);

    assert(blays[3]->ndim == 3);
    assert(blays[3]->shape[0] == 3);
    assert(blays[3]->shape[1] == 2);
    assert(blays[3]->shape[2] == 4);
    assert(blays[3]->stride[0] == 1);
    assert(blays[3]->stride[1] == 0);
    assert(blays[3]->stride[2] == 0);

    layout_free(lay1);
    layout_free(lay2);
    layout_free(lay3);
    layout_free(lay4);
    for (size_t i = 0; i < 4; i++) {
        layout_free(blays[i]);
    }
    free(blays);
}

int main() {
    test_alloc();
    test_utils();
    test_array_creation();
    test_iterator();
    test_array_initialization();
    test_array_operations();
    test_reduction_operations();
    test_view_from_range();
    test_modify_view();
    test_non_contiguous_view();
    test_basic_matmul();
    test_matmul_identity();
    test_blocked_matmul();
    test_array_reduce_max();
    test_array_reduce_sum();
    test_array_reshape();
    test_move_dim();
    test_transpose();
    test_dimension_mismatch();
    test_stack();
    test_layout_broadcast_multiple();
    assert(assertion_failures == 2);

    printf("tests passed %d.", assertion_failures);
    return 0;
}